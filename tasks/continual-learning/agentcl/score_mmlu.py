"""MMLU-Pro answer scorer, run as each task's verifier.

One question, one option letter: reward 1.0 for the labelled answer and
0.0 for anything else, which is the accuracy AgentCL reports for its
MMLU-Pro streams. Parsing follows upstream's ``parse_answer``: a JSON
object with an ``answer`` key first, then the two anchored fallbacks, so
an agent that writes ``{"answer": "G"}`` and one that writes ``G`` are
not scored differently for their formatting.

Every fallback is anchored to one letter the submission *names*. Nothing
here scans the file for a letter that happens to match the key, so a
submission listing every option earns 0, not a free point.

Deterministic and offline.
"""

import json
import os
import re

ANSWER_PATH = "/app/answer.json"
TRUTH_PATH = "/tests/truth.json"
REWARD_PATH = "/logs/verifier/reward.txt"
DETAILS_PATH = "/logs/verifier/scoring.json"

# A submitted answer is one letter. Anything past this is not an answer,
# and reading it in full would only feed the parser noise.
MAX_SUBMISSION_BYTES = 64 * 1024

_JSON_ANSWER = re.compile(r'"answer"\s*:\s*"?([A-Za-z])"?')
_BARE_LETTER = re.compile(r"^[A-Za-z]$")


def parse_answer(raw: str, labels: list[str]) -> str | None:
    """The option letter a submission names, or None if it names none.

    Ported from AgentCL's ``mmlu_runner.parse_answer`` minus the pieces
    that only a chat transcript needs: the declared JSON object, its
    first-match regex shadow for near-JSON, and a file holding just the
    letter. A letter outside *labels* is not an answer.
    """
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        parsed = None
    if isinstance(parsed, dict):
        answer = str(parsed.get("answer", parsed.get("Answer", ""))).strip().upper()
        return answer if answer in labels else None
    match = _JSON_ANSWER.search(raw)
    if match:
        answer = match.group(1).upper()
        return answer if answer in labels else None
    stripped = raw.strip().strip("\"'").strip()
    if _BARE_LETTER.match(stripped) and stripped.upper() in labels:
        return stripped.upper()
    return None


def read_submission(path: str) -> tuple[str | None, str]:
    """The agent's answer file as text; a missing file is not an error."""
    if not os.path.exists(path):
        return None, f"no answer file at {path}"
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read(MAX_SUBMISSION_BYTES), "ok"
    except OSError as error:
        return None, f"answer file unreadable: {error}"


def grade(raw: str | None, truth: dict) -> dict:
    """Score one submission against one question's key.

    *raw* is the submitted text, or None when nothing was submitted.
    Every zero carries a reason, so a run that scores 0 across a stream
    can be told apart from a run that never wrote the file.
    """
    labels = [str(label).upper() for label in truth["labels"]]
    correct = str(truth["answer"]).upper()
    if correct not in labels:
        raise ValueError(f"answer key {correct!r} is not one of {labels}")
    answer = parse_answer(raw, labels) if raw is not None else None
    if answer is None:
        return {
            "score": 0.0,
            "answer": None,
            "correct_answer": correct,
            "reason": "submission names no listed option letter",
        }
    if answer != correct:
        return {
            "score": 0.0,
            "answer": answer,
            "correct_answer": correct,
            "reason": f"answered {answer}, the correct option was {correct}",
        }
    return {
        "score": 1.0,
        "answer": answer,
        "correct_answer": correct,
        "reason": "correct",
    }


def main() -> int:
    os.makedirs(os.path.dirname(REWARD_PATH), exist_ok=True)
    with open(TRUTH_PATH) as handle:
        truth = json.load(handle)
    raw, status = read_submission(ANSWER_PATH)
    result = {**grade(raw, truth), "submission_status": status}
    with open(REWARD_PATH, "w") as handle:
        handle.write(str(result["score"]))
    with open(DETAILS_PATH, "w") as handle:
        json.dump(result, handle, indent=1)
    print(f"mmlu-pro accuracy: {result['score']} ({result['reason']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
