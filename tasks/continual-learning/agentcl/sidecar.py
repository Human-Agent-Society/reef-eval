"""The judge-sidecar scaffolding every interactive AgentCL task shares.

Four of the five subsets hide something from the agent -- the tests, the
corpus, the environment -- behind a judge container it reaches over
HTTP. The Harbor wiring for that is the same every time: a compose
overlay that starts the judge and hands ``main`` its URL, an agent image
with curl, a judge image built from the subset's own Dockerfile, and a
verifier that asks the judge for the reward. Only the server and the
instance differ, so only those are arguments.
"""

import json
import shutil
from pathlib import Path

HERE = Path(__file__).parent

AGENT_DOCKERFILE = """FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl \\
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
"""

COMPOSE = """services:
  main:
    depends_on:
      judge:
        condition: service_healthy
    environment:
      JUDGE_URL: "http://judge:8082"

  judge:
    build:
      context: ${{CONTEXT_DIR}}
      dockerfile: Dockerfile.judge
    command: ["python3", "/judge/{server}"]
    networks: [default]{volumes}
    environment:
      PORT: "8082"
      JUDGE_DIR: "/judge"
      PYTHONPATH: "/judge"
    healthcheck:
      test: ["CMD", "python3", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8082/health', timeout=2)"]
      interval: 5s
      timeout: 3s
      retries: 60
      start_period: 3s
"""

# The judge owns the score, so the verifier only has to ask for it.
TEST_SH = """#!/bin/bash
mkdir -p /logs/verifier
curl -s "$JUDGE_URL/final" > /logs/verifier/final.json
python3 -c "import json; print(json.load(open('/logs/verifier/final.json'))['reward'])" > /logs/verifier/reward.txt
"""


def scaffold(
    task_dir: Path,
    *,
    server: str,
    judge_dockerfile: str,
    instance: dict,
    agent_dockerfile: str = AGENT_DOCKERFILE,
    extra_files: tuple[str, ...] = (),
    judge_volumes: tuple[str, ...] = (),
) -> None:
    """Write everything a sidecar task needs except its instruction.

    *instance* is the hidden half of the task -- the tests, the answer,
    the environment spec. It lands in the build context both images
    share, and only ``Dockerfile.judge`` copies from there, which is what
    keeps it out of the container the agent runs in.

    *judge_volumes* are compose volume lines for data too large to bake
    into an image, such as the BrowseComp+ corpus.
    """
    environment = task_dir / "environment"
    environment.mkdir(parents=True, exist_ok=True)
    (task_dir / "tests").mkdir(exist_ok=True)
    (task_dir / "solution").mkdir(exist_ok=True)

    (environment / "Dockerfile").write_text(agent_dockerfile)
    (environment / "Dockerfile.judge").write_text(judge_dockerfile)
    volumes = (
        "\n    volumes:\n" + "\n".join(f'      - "{line}"' for line in judge_volumes)
        if judge_volumes
        else ""
    )
    (environment / "docker-compose.yaml").write_text(
        COMPOSE.format(server=server, volumes=volumes)
    )
    for name in ("judge_http.py", server, *extra_files):
        shutil.copy(HERE / name, environment / name)
    (environment / "instance.json").write_text(json.dumps(instance, indent=1))
    (task_dir / "tests" / "test.sh").write_text(TEST_SH)
