"""AgentCL conversion, scorers and judges: real fixture rows, known answers.

AgentCL is CC-BY-NC-4.0, so its converted tasks are never committed to
this repo and `tests/test_task_suite.py` never sees them. Everything that
suite would enforce for a committed task is enforced here instead,
against tasks built in a tmp dir: stock-Harbor validation, the reference
solution scoring what its task claims, and the zero-reward cases that try
to game the grader.

The judges get the same treatment. CodeEval-Pro's and BrowseComp+'s run
here exactly as they would in their container; the two that wrap a
simulator run against a stub of it, which still holds the protocol, the
step budget and the progress accounting in place.

Fixtures are small slices of the pinned files, quoted for testing.
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


sys.path.insert(0, str(AGENTCL))  # the judges import judge_http as a sibling

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


# ================================================================ MMLU-Pro


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


# ============================================================ shared sidecar


def _load_judge(name: str, task_dir: Path):
    """Import a judge server as its container would, for one task."""
    import os

    os.environ["JUDGE_DIR"] = str(task_dir / "environment")
    sys.modules.pop(f"agentcl_{name}", None)
    sys.modules.pop(name, None)
    return _load(name)


def _dockerfiles(task_dir: Path) -> tuple[str, str]:
    environment = task_dir / "environment"
    return (
        (environment / "Dockerfile").read_text(),
        (environment / "Dockerfile.judge").read_text(),
    )


def _assert_sidecar_shape(task_dir: Path, server: str) -> None:
    for piece in (
        "task.toml",
        "instruction.md",
        "environment/Dockerfile",
        "environment/Dockerfile.judge",
        "environment/docker-compose.yaml",
        "environment/judge_http.py",
        f"environment/{server}",
        "environment/instance.json",
        "tests/test.sh",
        "solution/solve.sh",
    ):
        assert (task_dir / piece).exists(), f"{task_dir.name} missing {piece}"
    compose = (task_dir / "environment" / "docker-compose.yaml").read_text()
    assert f'"/judge/{server}"' in compose
    assert 'JUDGE_URL: "http://judge:8082"' in compose
    assert "healthcheck" in compose  # main waits for the judge to answer
    agent_dockerfile, judge_dockerfile = _dockerfiles(task_dir)
    # The hidden half reaches the judge's image and no other.
    assert "instance.json" in judge_dockerfile
    assert "instance.json" not in agent_dockerfile
    assert "/logs/verifier/reward.txt" in (task_dir / "tests" / "test.sh").read_text()


# ======================================================== CodeEval-Pro


codeeval = _load("convert_codeeval")
CODE_ROWS = json.loads((FIXTURES / "codeeval_humaneval_sample.json").read_text())
CODE_BY_ID = {row["id"]: row for row in CODE_ROWS}


def _convert_code(tmp_path: Path, rows=None) -> list[Path]:
    return codeeval.convert_stream(
        rows if rows is not None else CODE_ROWS,
        tmp_path,
        dataset="humaneval",
        stream="naive",
    )


def test_codeeval_writes_a_sidecar_task(tmp_path):
    tasks = _convert_code(tmp_path)
    assert [task.name for task in tasks] == [
        "code-humaneval-naive-001",
        "code-humaneval-naive-002",
        "code-humaneval-naive-003",
    ]
    for task in tasks:
        _assert_sidecar_shape(task, "codeeval_server.py")
        assert (task / "environment" / "submit.py").exists()


def test_codeeval_hides_the_tests_and_the_reference(tmp_path):
    """The prompt is the prefix. The tests, the answer and the reference
    are in the instance, which only the judge's image copies."""
    for task, row in zip(_convert_code(tmp_path), CODE_ROWS, strict=True):
        instruction = (task / "instruction.md").read_text()
        instance = json.loads((task / "environment" / "instance.json").read_text())
        assert instance["test_code"] == row["test_code"]
        assert row["test_code"].strip()[:60] not in instruction
        assert instance["prefix"].rstrip() in instruction
        _, reference = codeeval.split_prefix(row)
        body = reference.strip().splitlines()
        assert body and body[-1].strip() not in instruction


def test_codeeval_repairs_the_truncated_prompts(tmp_path):
    """Three HumanEval rows put the tail of the prompt in `solution`. The
    split has to land on the signature, or the prompt leaks the answer."""
    row = CODE_BY_ID[46]
    assert "def " not in row["problem"]  # the fixture's truncated row
    prefix, reference = codeeval.split_prefix(row)
    assert prefix.startswith(row["problem"].rstrip())
    assert prefix.rstrip().endswith(":")  # ends on the signature line
    assert row["solution"].endswith(reference)  # the body, and only the body
    instruction = (_convert_code(tmp_path, [row])[0] / "instruction.md").read_text()
    for line in reference.strip().splitlines():
        assert line not in instruction


def test_codeeval_records_the_ceiling_it_can_actually_reach(tmp_path):
    """Seven released problems ship a reference that cannot pass their own
    tests. Those tasks say oracle_score 0.0 rather than claiming 1.0."""
    tasks = _convert_code(tmp_path)
    good = tomllib.loads((tasks[0] / "task.toml").read_text())["metadata"]
    broken = tomllib.loads((tasks[1] / "task.toml").read_text())["metadata"]
    assert good["problem_id"] == "humaneval-new-91" and good["oracle_score"] == 1.0
    assert broken["problem_id"] == "humaneval-new-0" and broken["oracle_score"] == 0.0
    assert "typing" in broken["reference_note"]
    assert ("humaneval", "new", "0") in codeeval.REFERENCE_FAILURES


def _submit(judge, completion):
    return judge.submit({"completion": completion})


def test_codeeval_judge_scores_the_reference(tmp_path):
    task = _convert_code(tmp_path, [CODE_BY_ID[91]])[0]
    judge = _load_judge("codeeval_server", task)
    _, reference = codeeval.split_prefix(CODE_BY_ID[91])
    payload, status = _submit(judge, reference)
    assert status == 200 and payload["passed"], payload["feedback"]
    assert judge.final({})["reward"] == 1.0


def test_codeeval_judge_reports_failures_and_meters_them(tmp_path):
    task = _convert_code(tmp_path, [CODE_BY_ID[91]])[0]
    judge = _load_judge("codeeval_server", task)
    payload, status = _submit(judge, "return 0")
    assert status == 200 and not payload["passed"]
    assert (
        "AssertionError" in payload["feedback"] or "Tests failed" in payload["feedback"]
    )
    assert payload["attempts_remaining"] == judge.MAX_ATTEMPTS - 1
    for _ in range(judge.MAX_ATTEMPTS - 1):
        _submit(judge, "return 0")
    exhausted, status = _submit(judge, "return 0")
    assert status == 429 and exhausted["reward"] == 0.0
    assert judge.final({})["reward"] == 0.0


def test_codeeval_judge_refuses_to_keep_scoring_after_a_pass(tmp_path):
    task = _convert_code(tmp_path, [CODE_BY_ID[91]])[0]
    judge = _load_judge("codeeval_server", task)
    _, reference = codeeval.split_prefix(CODE_BY_ID[91])
    _submit(judge, reference)
    payload, status = _submit(judge, "return 0")
    assert status == 409 and payload["reward"] == 1.0  # a pass cannot be undone


def test_codeeval_judge_takes_an_unindented_body(tmp_path):
    """Upstream re-indents a flush-left completion rather than failing it."""
    task = _convert_code(tmp_path, [CODE_BY_ID[91]])[0]
    judge = _load_judge("codeeval_server", task)
    _, reference = codeeval.split_prefix(CODE_BY_ID[91])
    flush_left = "\n".join(line[4:] for line in reference.splitlines())
    payload, _ = _submit(judge, flush_left)
    assert payload["passed"], payload["feedback"]


def test_codeeval_judge_is_empty_handed_without_a_submission(tmp_path):
    task = _convert_code(tmp_path, [CODE_BY_ID[91]])[0]
    judge = _load_judge("codeeval_server", task)
    result = judge.final({})
    assert result["reward"] == 0.0 and result["note"] == "no submission"


def test_the_broken_reference_really_is_broken(tmp_path):
    """The teeth behind REFERENCE_FAILURES: if upstream ever fixes this
    row, the table is stale and this test says so."""
    row = CODE_BY_ID[0]
    task = _convert_code(tmp_path, [row])[0]
    judge = _load_judge("codeeval_server", task)
    _, reference = codeeval.split_prefix(row)
    payload, _ = _submit(judge, reference)
    assert not payload["passed"]
    assert "NameError" in payload["feedback"]


# ============================================================ BrowseComp+


browsecomp = _load("convert_browsecomp")
corpus_module = _load("browsecomp_corpus")
BROWSE_ROWS = [
    json.loads(line)
    for line in (FIXTURES / "browsecomp_sample.jsonl").read_text().splitlines()
    if line
]


def _tiny_corpus(tmp_path: Path) -> Path:
    """Three documents, one of which answers the fixture's first query."""
    index = tmp_path / "corpus.sqlite"
    documents = [
        ("29830", "Lush Life (TV series)", "Lush Life was a short-lived 1996 sitcom."),
        ("41163", "Karen Duffy", "Duffy is an American actress and television host."),
        ("62932", "Unrelated", "A document about the migration of arctic terns."),
    ]
    corpus_module.build_index(iter(documents), index)
    return index


def test_browsecomp_writes_a_sidecar_task_that_mounts_the_corpus(tmp_path):
    tasks = browsecomp.convert_stream(BROWSE_ROWS, tmp_path, stream="naive")
    assert [task.name for task in tasks][:2] == [
        "browsecomp-naive-001",
        "browsecomp-naive-002",
    ]
    _assert_sidecar_shape(tasks[0], "browsecomp_server.py")
    compose = (tasks[0] / "environment" / "docker-compose.yaml").read_text()
    assert f"${{{browsecomp.CORPUS_ENV_VAR}}}:/judge/corpus.sqlite:ro" in compose
    assert (tasks[0] / "environment" / "browsecomp_corpus.py").exists()


def test_browsecomp_keeps_the_answer_out_of_what_we_wrote(tmp_path):
    """The answer reaches the judge and nothing else we generate.

    It is checked against the instruction minus the question, because a
    few upstream subqueries name their own answer inside the question
    ("in the article 'Lush Life (TV series)', which series ...") — that
    is the released wording, and repairing it would be a different
    benchmark.
    """
    for task, row in zip(
        browsecomp.convert_stream(BROWSE_ROWS, tmp_path, stream="naive"),
        BROWSE_ROWS,
        strict=True,
    ):
        instruction = (task / "instruction.md").read_text()
        assert row["query"] in instruction
        assert row["answer"] not in instruction.replace(row["query"], "")
        assert (
            json.loads((task / "environment" / "instance.json").read_text())["answer"]
            == row["answer"]
        )


def test_browsecomp_stream_shape_is_checked(tmp_path):
    """`comp` is the subqueries followed by the originals; a file that is
    not the released one stops the conversion."""
    (tmp_path / "ground_truth_original.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in BROWSE_ROWS[:2])
    )
    with pytest.raises(ValueError, match="100 rows"):
        browsecomp.load_stream(tmp_path, "naive")


def test_the_corpus_index_finds_documents_and_opens_them(tmp_path):
    index = _tiny_corpus(tmp_path)
    corpus = corpus_module.BrowseCompCorpus(index)
    try:
        assert corpus.count() == 3
        hits = corpus.search("short-lived sitcom", limit=5)
        assert hits and hits[0]["doc_id"] == "29830"
        assert corpus.open("29830")["title"] == "Lush Life (TV series)"
        assert corpus.open("nope") is None
        assert corpus.missing_doc_ids(["29830", "absent"]) == ["absent"]
    finally:
        corpus.close()


def _browsecomp_judge(tmp_path: Path, row: dict):
    task = browsecomp.convert_stream([row], tmp_path / "task", stream="naive")[0]
    import os

    os.environ["CORPUS_PATH"] = str(_tiny_corpus(tmp_path))
    return _load_judge("browsecomp_server", task)


def test_browsecomp_judge_grades_the_published_answer(tmp_path):
    row = BROWSE_ROWS[0]
    judge = _browsecomp_judge(tmp_path, row)
    results, status = judge.search({"query": "short-lived sitcom"})
    assert status == 200 and results["results"]
    assert results["calls_remaining"] == judge.MAX_TOOL_CALLS - 1
    payload, status = judge.answer({"answer": row["answer"]})
    assert status == 200 and payload["reward"] == 1.0
    assert judge.final({})["reward"] == 1.0


@pytest.mark.parametrize(
    ("submitted", "reward"),
    [
        ("lush life", 1.0),  # casefolded
        ("  Lush  Life!  ", 1.0),  # punctuation and spacing collapsed
        ("Lush Life is the series", 0.0),  # a sentence is not the answer
        ("", 0.0),
    ],
)
def test_browsecomp_judge_uses_upstream_normalisation(submitted, reward, tmp_path):
    row = {**BROWSE_ROWS[0], "answer": "Lush Life"}
    judge = _browsecomp_judge(tmp_path, row)
    payload, _ = judge.answer({"answer": submitted})
    assert payload["reward"] == reward


def test_browsecomp_judge_meters_tools_and_ends_on_the_answer(tmp_path):
    judge = _browsecomp_judge(tmp_path, BROWSE_ROWS[0])
    for _ in range(judge.MAX_TOOL_CALLS):
        judge.search({"query": "sitcom"})
    refused, status = judge.search({"query": "sitcom"})
    assert status == 429 and "budget" in refused["error"]
    judge.answer({"answer": "nope"})
    _, status = judge.search({"query": "sitcom"})
    assert status == 409  # answering is terminal
    assert judge.final({})["reward"] == 0.0


def test_browsecomp_judge_says_when_nothing_was_answered(tmp_path):
    judge = _browsecomp_judge(tmp_path, BROWSE_ROWS[0])
    result = judge.final({})
    assert result["reward"] == 0.0 and result["note"] == "no answer submitted"


# ============================================================ AgentBoard envs


babyai = _load("convert_babyai")
scienceworld = _load("convert_scienceworld")
BABYAI_ROWS = [
    json.loads(line)
    for line in (FIXTURES / "babyai_sample.jsonl").read_text().splitlines()
    if line
]
BABYAI_SEEDS = babyai.load_seeds(FIXTURES / "babyai_seeds.json")
SCIWORLD_ROWS = [
    json.loads(line)
    for line in (FIXTURES / "scienceworld_sample.jsonl").read_text().splitlines()
    if line
]


class _StubBabyAI:
    """Enough of AgentBoard's BabyAI to hold the protocol in place."""

    def __init__(self, game_name, seed, max_episode_steps, obs_to_reward):
        self.game_name = game_name
        self.seed = seed
        self.obs_to_reward = list(obs_to_reward)
        self.init_obs = _StubBabyAI.init_observation
        self.reward = 0.0
        self._obs = self.init_obs

    def reset(self):
        return self.init_obs

    def _get_obs(self):
        return self._obs

    def _get_action_space(self):
        return ["go to the red ball", "pick up the red ball"]

    def step(self, action):
        self._obs = f"You did: {action}"
        if action == "go to the red ball":
            self.reward = 0.5
        if action == "pick up the red ball":
            self.reward = 1.0
        return self._obs, self.reward, self.reward >= 1.0, {"action": action}

    def close(self):
        pass


def _install_babyai_stub(monkeypatch, init_obs: str):
    import types

    _StubBabyAI.init_observation = init_obs
    module = types.ModuleType("agentenv_babyai")
    environment = types.ModuleType("agentenv_babyai.environment")
    environment.BabyAI = _StubBabyAI
    module.environment = environment
    monkeypatch.setitem(sys.modules, "agentenv_babyai", module)
    monkeypatch.setitem(sys.modules, "agentenv_babyai.environment", environment)


def test_babyai_writes_a_sidecar_task_with_the_seed_hidden(tmp_path):
    tasks = babyai.convert_stream(BABYAI_ROWS, BABYAI_SEEDS, tmp_path, stream="naive")
    assert [task.name for task in tasks] == ["babyai-naive-001", "babyai-naive-002"]
    _assert_sidecar_shape(tasks[0], "babyai_server.py")
    instance = json.loads((tasks[0] / "environment" / "instance.json").read_text())
    assert instance["subgoals"] == BABYAI_ROWS[0]["subgoals"]
    assert instance["seed"] == BABYAI_SEEDS[BABYAI_ROWS[0]["id"]]
    instruction = (tasks[0] / "instruction.md").read_text()
    assert BABYAI_ROWS[0]["goal"] in instruction
    for subgoal in BABYAI_ROWS[0]["subgoals"]:
        assert subgoal not in instruction  # the checkpoints stay hidden
    assert "AgentGym" not in instruction


def test_babyai_refuses_an_episode_with_no_published_seed(tmp_path):
    with pytest.raises(ValueError, match="seed manifest"):
        babyai.convert_episode(
            BABYAI_ROWS[0], {}, tmp_path, stream="naive", position=0, total=1
        )


def test_babyai_judge_steps_scores_and_stops(tmp_path, monkeypatch):
    row = BABYAI_ROWS[0]
    _install_babyai_stub(monkeypatch, row["additional_info"]["init_obs"])
    task = babyai.convert_stream([row], BABYAI_SEEDS, tmp_path, stream="naive")[0]
    judge = _load_judge("babyai_server", task)

    view = judge.observe({})
    assert view["goal"] == row["goal"] and view["steps_remaining"] == judge.MAX_STEPS
    half, status = judge.step({"action": "go to the red ball"})
    assert status == 200 and half["progress"] == 0.5
    assert judge.final({})["reward"] == 0.5  # PR, not a pass/fail
    done, _ = judge.step({"action": "pick up the red ball"})
    assert done["done"] and judge.final({})["success"] is True
    refused, status = judge.step({"action": "go to the red ball"})
    assert status == 409


def test_babyai_judge_will_not_grade_a_different_episode(tmp_path, monkeypatch):
    """The seed has to reproduce the published opening observation."""
    row = BABYAI_ROWS[0]
    _install_babyai_stub(monkeypatch, "a different room entirely")
    task = babyai.convert_stream([row], BABYAI_SEEDS, tmp_path, stream="naive")[0]
    judge = _load_judge("babyai_server", task)
    with pytest.raises(RuntimeError, match="seed mismatch"):
        judge.observe({})


def test_babyai_judge_stops_at_the_step_budget(tmp_path, monkeypatch):
    row = BABYAI_ROWS[0]
    _install_babyai_stub(monkeypatch, row["additional_info"]["init_obs"])
    task = babyai.convert_stream([row], BABYAI_SEEDS, tmp_path, stream="naive")[0]
    judge = _load_judge("babyai_server", task)
    for _ in range(judge.MAX_STEPS):
        judge.step({"action": "look"})
    over, status = judge.step({"action": "look"})
    assert status == 409 and over["steps_remaining"] == 0
    assert over["note"] == "episode is over"


class _StubScienceWorld:
    """Enough of ScienceWorld to hold the protocol and the scoring."""

    def __init__(self):
        self.loaded = None
        self.score = 0.0
        self.done = False

    def load(self, env_name, var, simplificationStr):  # noqa: N803 - upstream name
        self.loaded = (env_name, var, simplificationStr)

    def reset(self):
        return "This room is the hallway.", {}

    def inventory(self):
        return "In your inventory, you see: nothing"

    def step(self, action):
        if action == "go to outside":
            return "You move to the outside.", 0.0, False, {"score": 50.0}
        if action == "focus on crocodile egg":
            return "You focus on the crocodile egg.", 1.0, True, {"score": 100.0}
        return f"You did: {action}", 0.0, False, {"score": self.score}

    def getPossibleActions(self):  # noqa: N802 - upstream name
        return ["go to OBJ", "focus on OBJ"]

    def getPossibleObjects(self):  # noqa: N802 - upstream name
        return ["outside", "crocodile egg"]

    def close(self):
        pass


def _install_scienceworld_stub(monkeypatch):
    import types

    module = types.ModuleType("scienceworld")
    module.ScienceWorldEnv = _StubScienceWorld
    monkeypatch.setitem(sys.modules, "scienceworld", module)


def test_scienceworld_writes_a_sidecar_task_with_the_subgoals_hidden(tmp_path):
    tasks = scienceworld.convert_stream(SCIWORLD_ROWS, tmp_path, stream="naive")
    assert [task.name for task in tasks] == [
        "sciworld-naive-001",
        "sciworld-naive-002",
    ]
    _assert_sidecar_shape(tasks[0], "scienceworld_server.py")
    instance = json.loads((tasks[0] / "environment" / "instance.json").read_text())
    assert instance["env_name"] == SCIWORLD_ROWS[0]["additional_info"]["env_name"]
    instruction = (tasks[0] / "instruction.md").read_text()
    assert SCIWORLD_ROWS[0]["goal"] in instruction
    for subgoal in SCIWORLD_ROWS[0]["subgoals"]:
        assert subgoal not in instruction
    metadata = tomllib.loads((tasks[0] / "task.toml").read_text())["metadata"]
    assert metadata["stream"] == "naive" and metadata["position"] == 0
    assert metadata["reference"] == "pipeline-check"  # no gold trajectory ships


def test_scienceworld_judge_scores_progress_and_success_apart(tmp_path, monkeypatch):
    _install_scienceworld_stub(monkeypatch)
    row = SCIWORLD_ROWS[0]
    task = scienceworld.convert_stream([row], tmp_path, stream="naive")[0]
    judge = _load_judge("scienceworld_server", task)

    view = judge.observe({})
    assert "hallway" in view["observation"] and "inventory" in view["observation"]
    assert judge.actions({})["possible_objects"] == ["outside", "crocodile egg"]
    half, _ = judge.step({"action": "go to outside"})
    assert half["progress"] == 0.5  # one of the row's two subgoals matched
    full, _ = judge.step({"action": "focus on crocodile egg"})
    assert full["progress"] == 1.0 and full["done"]
    result = judge.final({})
    assert result["reward"] == 1.0 and result["success"] is True
    assert result["native_score"] == 100.0


def test_scienceworld_judge_refuses_the_navigation_shortcut(tmp_path, monkeypatch):
    """`teleport` would skip the part of the task being measured."""
    _install_scienceworld_stub(monkeypatch)
    task = scienceworld.convert_stream([SCIWORLD_ROWS[0]], tmp_path, stream="naive")[0]
    judge = _load_judge("scienceworld_server", task)
    payload, status = judge.step({"action": "teleport to outside"})
    assert status == 200
    assert payload["observation"] == "No known action matches that input."
    assert payload["progress"] == 0.0  # and it still costs a step
    assert payload["steps_used"] == 1


def test_scienceworld_judge_loads_the_agentboard_simplifications(tmp_path, monkeypatch):
    _install_scienceworld_stub(monkeypatch)
    task = scienceworld.convert_stream([SCIWORLD_ROWS[0]], tmp_path, stream="naive")[0]
    judge = _load_judge("scienceworld_server", task)
    env = judge.environment()
    assert env.loaded[2] == judge.AGENTBOARD_SIMPLIFICATIONS


# ============================================================ every subset


ALL_CONVERTERS = (
    "convert_mmlu",
    "convert_codeeval",
    "convert_browsecomp",
    "convert_babyai",
    "convert_scienceworld",
)


def _one_task_per_subset(tmp_path: Path) -> list[Path]:
    return [
        _convert(tmp_path / "mmlu")[0],
        _convert_code(tmp_path / "code", [CODE_BY_ID[91]])[0],
        browsecomp.convert_stream(BROWSE_ROWS[:1], tmp_path / "bc", stream="naive")[0],
        babyai.convert_stream(
            BABYAI_ROWS[:1], BABYAI_SEEDS, tmp_path / "ba", stream="naive"
        )[0],
        scienceworld.convert_stream(SCIWORLD_ROWS[:1], tmp_path / "sw", stream="naive")[
            0
        ],
    ]


def test_every_subset_converts_to_valid_stock_harbor(tmp_path):
    pytest.importorskip("harbor")
    from harbor.models.task.config import TaskConfig

    for task in _one_task_per_subset(tmp_path):
        TaskConfig.model_validate(tomllib.loads((task / "task.toml").read_text()))


def test_every_task_names_its_benchmark_and_its_place_in_the_stream(tmp_path):
    for task in _one_task_per_subset(tmp_path):
        metadata = tomllib.loads((task / "task.toml").read_text())["metadata"]
        assert metadata["benchmark"] == "agentcl"
        assert metadata["subset"]
        assert metadata["position"] == 0
        assert "CC-BY-NC-4.0" in metadata["license_note"]


def test_every_task_runs_offline(tmp_path):
    """Every subset here is public data, so egress is a lookup channel."""
    for task in _one_task_per_subset(tmp_path):
        agent = tomllib.loads((task / "task.toml").read_text())["agent"]
        assert agent["network_mode"] == "allowlist"
        assert agent.get("allowed_hosts", []) in ([], ["judge"])


# ============================================================ the judge's HTTP


judge_http = _load("judge_http")


@pytest.fixture
def judge_server():
    """The shared judge plumbing, serving on a loopback port."""
    import socket
    import threading
    import urllib.error
    import urllib.request

    def echo(body):
        return {"saw": body}

    def refuse(_body):
        return {"error": "no"}, 429

    def explode(_body):
        raise RuntimeError("the simulator fell over")

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    routes = {
        ("POST", "/echo"): echo,
        ("POST", "/refuse"): refuse,
        ("GET", "/explode"): explode,
    }
    thread = threading.Thread(target=judge_http.serve, args=(routes, port), daemon=True)
    thread.start()

    def call(method, path, body=None):
        url = f"http://127.0.0.1:{port}{path}"
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as error:
            return error.code, json.load(error)

    for _ in range(50):  # the server binds asynchronously
        try:
            call("GET", "/health")
            break
        except Exception:  # noqa: BLE001 - still starting
            import time

            time.sleep(0.02)
    return call


def test_the_judge_answers_health_before_anything_else(judge_server):
    """Compose waits on /health, so `main` never races the judge."""
    assert judge_server("GET", "/health") == (200, {"ok": True})


def test_the_judge_routes_and_reports(judge_server):
    assert judge_server("POST", "/echo", {"a": 1}) == (200, {"saw": {"a": 1}})
    status, payload = judge_server("POST", "/refuse", {})
    assert status == 429 and payload["error"] == "no"
    status, payload = judge_server("POST", "/nowhere", {})
    assert status == 404 and "unknown endpoint" in payload["error"]


def test_a_judge_that_breaks_says_so_instead_of_hanging(judge_server):
    status, payload = judge_server("GET", "/explode")
    assert status == 500 and "the simulator fell over" in payload["error"]


def test_the_judge_rejects_a_body_that_is_not_an_object(judge_server):
    status, payload = judge_server("POST", "/echo", ["not", "an", "object"])
    assert status == 400 and "JSON object" in payload["error"]


def test_the_babyai_seed_manifest_ships_with_the_converter():
    """AgentCL publishes the opening observation, not the seed behind it.

    The manifest is ours, so it is checked here for shape and coverage —
    and at run time by the judge, which rebuilds the room and refuses to
    serve if the observation is not the published one.
    """
    seeds = babyai.load_seeds(babyai.SEEDS)
    assert len(seeds) == 76  # the episodes the two streams draw on
    assert all(isinstance(key, int) and value >= 0 for key, value in seeds.items())
    for row in BABYAI_ROWS:
        assert int(row["id"]) in seeds
