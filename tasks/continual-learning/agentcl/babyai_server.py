"""BabyAI judge: the AgentBoard environment for one AgentCL task.

Runs in the judge container so the environment, its subgoals and the
score stay out of the agent's reach; the agent only sends actions. The
instance is the released row: the AgentBoard subtask, the seed AgentCL
publishes for it, and the subgoal list that BabyAI turns into progress
(``obs_to_reward``). If the environment's opening observation is not the
one the row recorded, the judge refuses to serve rather than quietly
grading a different episode.

Reward is progress -- the fraction of the row's subgoals reached -- which
is what AgentCL reports as PR; ``success`` records the environment's own
terminal completion alongside it.

Endpoints: ``GET /observe`` — ``POST /step`` {"action"} — ``GET /final``,
the verifier's verdict — ``GET /health``.
"""

import json
import os
import threading
from pathlib import Path

from judge_http import port_from_env, serve

JUDGE_DIR = Path(os.environ.get("JUDGE_DIR", Path(__file__).parent))
INSTANCE = json.loads((JUDGE_DIR / "instance.json").read_text())

#: The harness default AgentCL runs these streams with.
MAX_STEPS = 30
#: BabyAI's own episode cap, as AgentCL configures the environment.
MAX_EPISODE_STEPS = 50

_lock = threading.Lock()
_state = {"steps": 0, "score": 0.0, "done": False}
_env = None


def normalized(observation: str) -> str:
    return " ".join(str(observation).split())


def environment():
    """Build the episode once, checking it is the published one."""
    global _env
    if _env is not None:
        return _env
    from agentenv_babyai.environment import BabyAI

    env = BabyAI(
        game_name=INSTANCE["subtask"],
        seed=int(INSTANCE["seed"]),
        max_episode_steps=MAX_EPISODE_STEPS,
        obs_to_reward=list(INSTANCE["subgoals"]),
    )
    env.reset()
    expected = normalized(INSTANCE["init_obs"])
    actual = normalized(env.init_obs)
    if actual != expected:
        raise RuntimeError(
            f"BabyAI seed mismatch for source id {INSTANCE['source_id']}: "
            f"expected={expected!r}, actual={actual!r}"
        )
    _state["score"] = float(env.reward)
    _env = env
    return env


def render(env) -> dict:
    return {
        "goal": INSTANCE["goal"],
        "observation": env._get_obs(),
        "available_actions": env._get_action_space(),
        "steps_used": _state["steps"],
        "steps_remaining": MAX_STEPS - _state["steps"],
        "progress": _state["score"],
        "done": _state["done"],
    }


def observe(_body: dict) -> dict:
    with _lock:
        return render(environment())


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
            return {**render(env), "note": "episode is over"}, 409
        _state["steps"] += 1
        _, reward, done, info = env.step(action)
        _state["score"] = float(reward)
        _state["done"] = bool(done) or _state["steps"] >= MAX_STEPS
        return {**render(env), "info": info}, 200


def final(_body: dict) -> dict:
    with _lock:
        return {
            # AgentCL's PR: the fraction of subgoals this episode reached.
            "reward": round(float(_state["score"]), 6),
            "progress": round(float(_state["score"]), 6),
            "success": bool(_state["done"] and _state["score"] >= 1.0),
            "steps_used": _state["steps"],
            "source_id": INSTANCE["source_id"],
            "subtask": INSTANCE["subtask"],
        }


if __name__ == "__main__":
    environment()  # fail at startup, not on the agent's first action
    serve(
        {
            ("GET", "/observe"): observe,
            ("POST", "/step"): step,
            ("GET", "/final"): final,
        },
        port_from_env(),
    )
