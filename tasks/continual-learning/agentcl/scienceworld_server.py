"""ScienceWorld judge: the AgentBoard environment for one AgentCL task.

Runs in the judge container with the simulator, the subgoal patterns and
the score; the agent only sends text actions. Semantics follow AgentCL's
AgentBoard configuration: the same simplifications, ``teleport`` refused
(it would skip the navigation the task is about), and two numbers kept
apart -- progress, the fraction of the row's subgoal patterns that have
matched an observation, which is what AgentCL reports as PR, and the
simulator's own terminal completion, which is SR.

Endpoints: ``GET /observe`` — ``GET /actions``, the simulator's action
and object hints — ``POST /step`` {"action"} — ``GET /final``, the
verifier's verdict — ``GET /health``.
"""

import json
import os
import re
import threading
from pathlib import Path

from judge_http import port_from_env, serve

JUDGE_DIR = Path(os.environ.get("JUDGE_DIR", Path(__file__).parent))
INSTANCE = json.loads((JUDGE_DIR / "instance.json").read_text())

AGENTBOARD_SIMPLIFICATIONS = (
    "selfWateringFlowerPots,openContainers,openDoors,noElectricalAction"
)
#: The harness default AgentCL runs these streams with.
MAX_STEPS = 30
#: Navigation is part of the task, so the simulator's shortcut is refused.
FORBIDDEN_NAVIGATION_SHORTCUT = re.compile(r"^\s*teleport(?:\s|$)", re.IGNORECASE)

_lock = threading.Lock()
_state = {
    "steps": 0,
    "matched": [False] * len(INSTANCE["subgoals"]),
    "native_score": 0.0,
    "observation": "",
    "done": False,
    "success": False,
}
_env = None


def environment():
    """Load the episode once: one task is one ScienceWorld variation."""
    global _env
    if _env is not None:
        return _env
    from scienceworld import ScienceWorldEnv

    env = ScienceWorldEnv()
    env.load(
        INSTANCE["env_name"],
        int(INSTANCE["var"]),
        simplificationStr=AGENTBOARD_SIMPLIFICATIONS,
    )
    observation, _ = env.reset()
    inventory = env.inventory()
    _state["observation"] = f"{observation}\n{inventory}" if inventory else observation
    _env = env
    return env


def progress() -> float:
    total = len(_state["matched"])
    return sum(_state["matched"]) / total if total else 1.0


def match_subgoals(observation: str) -> None:
    for index, pattern in enumerate(INSTANCE["subgoals"]):
        if re.search(pattern, observation):
            _state["matched"][index] = True


def render() -> dict:
    return {
        "task_description": INSTANCE["goal"],
        "observation": _state["observation"],
        "steps_used": _state["steps"],
        "steps_remaining": MAX_STEPS - _state["steps"],
        "progress": round(progress(), 6),
        "done": _state["done"],
    }


def observe(_body: dict) -> dict:
    with _lock:
        environment()
        return render()


def actions(_body: dict) -> dict:
    with _lock:
        env = environment()
        return {
            "possible_actions": env.getPossibleActions(),
            "possible_objects": env.getPossibleObjects(),
        }


def step(body: dict) -> tuple[dict, int]:
    action = body.get("action")
    if not isinstance(action, str) or not action.strip():
        return {"error": "send {'action': '<command>'}"}, 400
    with _lock:
        env = environment()
        if _state["done"]:
            # Spending the last step ends the episode, so this covers the
            # budget too: there is no state where steps are gone but the
            # episode is not over.
            return {**render(), "note": "episode is over"}, 409
        _state["steps"] += 1
        if FORBIDDEN_NAVIGATION_SHORTCUT.match(action):
            observation, native_done, info = (
                "No known action matches that input.",
                False,
                {"invalid_action": True},
            )
        else:
            observation, _, native_done, info = env.step(action)
        match_subgoals(observation)
        _state["observation"] = observation
        _state["native_score"] = float(info.get("score", 0.0))
        _state["success"] = bool(native_done) and _state["native_score"] >= 100.0
        _state["done"] = bool(native_done) or _state["steps"] >= MAX_STEPS
        return {**render(), "info": info}, 200


def final(_body: dict) -> dict:
    with _lock:
        return {
            # AgentCL's PR is subgoal progress; SR is the simulator's own
            # completion, reported beside it rather than folded in.
            "reward": round(progress(), 6),
            "progress": round(progress(), 6),
            "success": _state["success"],
            "native_score": _state["native_score"],
            "steps_used": _state["steps"],
            "task_name": INSTANCE["env_name"],
            "var_num": int(INSTANCE["var"]),
        }


if __name__ == "__main__":
    environment()  # fail at startup, not on the agent's first action
    serve(
        {
            ("GET", "/observe"): observe,
            ("GET", "/actions"): actions,
            ("POST", "/step"): step,
            ("GET", "/final"): final,
        },
        port_from_env(),
    )
