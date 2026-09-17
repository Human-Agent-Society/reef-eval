"""CodeEval-Pro judge: the hidden tests and the attempt budget for one problem.

Runs in the judge container with the tests the agent never sees. The
semantics are AgentCL's ``codeeval_runner``: the submission is a
completion appended to the published prefix, the tests are the released
``test_code`` (with ``check(<entry_point>)`` appended when they define
one), a run that exits 0 passes, and the problem is worth 1.0 if any of
the ``MAX_ATTEMPTS`` submissions passes. Upstream retries with the
failure text in the prompt; here the same text comes back from
``/submit``, so the budget is what stops the agent from brute-forcing.

Endpoints: ``POST /submit`` {"completion"} — ``GET /final``, the
verifier's verdict — ``GET /health``.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from judge_http import port_from_env, serve

JUDGE_DIR = Path(os.environ.get("JUDGE_DIR", Path(__file__).parent))
INSTANCE = json.loads((JUDGE_DIR / "instance.json").read_text())

#: Upstream's ``--max-attempts`` default: ten tries with test feedback.
MAX_ATTEMPTS = 10
#: Upstream's per-run limit; the subprocess is killed two seconds later.
TIMEOUT_SECONDS = 30
#: How much of the failure output comes back, as upstream truncates it.
MAX_DIAGNOSTICS_CHARS = 4000

_lock = threading.Lock()
_state = {"attempts": 0, "passed": False, "done": False}


def entrypoint_name(prefix: str) -> str:
    """The function the tests call: the last signature in the prefix."""
    matches = re.findall(r"(?m)^\s*def\s+([A-Za-z_]\w*)\s*\(", prefix)
    if not matches:
        raise ValueError("problem prefix has no Python function signature")
    return matches[-1]


def indent_for_prefix(prefix: str, completion: str) -> str:
    """Re-indent a body that was written flush left, as upstream does.

    A completion is appended inside the prefix's function, so a model
    that returns an unindented body is fixed rather than failed: format
    is not what this benchmark measures.
    """
    lines = completion.replace("\t", "    ").rstrip().splitlines()
    if not lines:
        return "    pass\n"
    first = next((line for line in lines if line.strip()), "")
    if first[: len(first) - len(first.lstrip())]:
        return "\n".join(lines).rstrip() + "\n"
    padding = "    "
    return "\n".join(padding + line if line.strip() else line for line in lines) + "\n"


def build_test_program(completion: str) -> str:
    prefix = INSTANCE["prefix"]
    tests = str(INSTANCE["test_code"]).rstrip() + "\n"
    check_call = ""
    if re.search(r"(?m)^\s*def\s+check\s*\(\s*candidate\s*\)", tests):
        check_call = f"\ncheck({entrypoint_name(prefix)})\n"
    return prefix + indent_for_prefix(prefix, completion) + "\n" + tests + check_call


def run_tests(completion: str) -> tuple[bool, str]:
    """Run the hidden tests against *completion*; never raises."""
    with tempfile.TemporaryDirectory(prefix="agentcl_codeeval_") as workdir:
        script = Path(workdir) / "candidate.py"
        script.write_text(build_test_program(completion))
        try:
            result = subprocess.run(
                [sys.executable, "-I", str(script)],
                cwd=workdir,
                text=True,
                capture_output=True,
                timeout=TIMEOUT_SECONDS + 2,
                check=False,
                env={
                    "PATH": os.environ.get("PATH", ""),
                    "HOME": workdir,
                    "TMPDIR": workdir,
                    "MPLBACKEND": "Agg",
                    "MPLCONFIGDIR": workdir,
                    "PYTHONHASHSEED": "0",
                },
            )
        except subprocess.TimeoutExpired:
            return False, f"Tests timed out after {TIMEOUT_SECONDS} seconds."
    if result.returncode == 0:
        return True, "All tests passed."
    diagnostics = (result.stderr or result.stdout).strip()[-MAX_DIAGNOSTICS_CHARS:]
    return False, "Tests failed.\n" + diagnostics


def verdict() -> dict:
    return {
        "reward": 1.0 if _state["passed"] else 0.0,
        "passed": _state["passed"],
        "attempts_used": _state["attempts"],
        "problem_id": INSTANCE["problem_id"],
    }


def submit(body: dict) -> tuple[dict, int]:
    completion = body.get("completion")
    if not isinstance(completion, str) or not completion.strip():
        return {"error": "send {'completion': '<python>'}"}, 400
    with _lock:
        if _state["passed"]:
            return {**verdict(), "note": "already passed"}, 409
        if _state["attempts"] >= MAX_ATTEMPTS:
            return {**verdict(), "note": "attempt budget exhausted"}, 429
        _state["attempts"] += 1
        passed, feedback = run_tests(completion)
        _state["passed"] = passed
        return {
            "passed": passed,
            "feedback": feedback,
            "attempts_used": _state["attempts"],
            "attempts_remaining": MAX_ATTEMPTS - _state["attempts"],
        }, 200


def final(_body: dict) -> dict:
    with _lock:
        _state["done"] = True
        result = verdict()
    if not result["attempts_used"]:
        result["note"] = "no submission"
    return result


if __name__ == "__main__":
    serve({("POST", "/submit"): submit, ("GET", "/final"): final}, port_from_env())
