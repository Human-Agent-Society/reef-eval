# AgentCL

[AgentCL](https://huggingface.co/datasets/osunlp/AgentCL) (OSU NLP,
CC-BY-NC-4.0): task streams for measuring whether an agent gets better at
a domain as it works through it. Their framework is an agent plus a
memory method run over a stream; here that is an agent plus the carried
`$REEF_EVAL_STATE_DIR`, and the gain their tables report is
`metrics.transfer` against the same tasks run in isolation.

The dataset is non-commercial, and this repo is Apache-2.0, so **no
AgentCL task is committed here**. [`fetch.py`](fetch.py) builds them on
your machine, the way SWE-bench Verified's does for its own licensing
reason.

## What is converted

| Stream | Tasks | One question | Reward |
|---|---|---|---|
| `mmlu-economics-qNNN` | 100 | an MMLU-Pro economics item, 3-10 options | 1.0 for the keyed option, else 0 |
| `mmlu-engineering-qNNN` | 100 | the same for engineering | same |
| `mmlu-philosophy-qNNN` | 100 | the same for philosophy | same |

The other four AgentCL subsets (`codeeval-pro`, `browsecomp_plus`,
`agentboard_babyai`, `agentboard_scienceworld`) are tracked on the
[roadmap](https://github.com/Human-Agent-Society/reef-eval/issues/19).

```bash
reef-eval fetch agentcl                 # all 300; add a domain or --limit N
reef-eval stream tasks/continual-learning/agentcl/mmlu-economics-* --agent claude-code
```

Stream **one domain at a time**: upstream runs the three as separate
streams, so `reef-eval stream agentcl` would run a 300-task
concatenation of all three, which is a different experiment. The
isolated baseline for `metrics.transfer` is `reef-eval run` over the
same task list.

## How the conversion works

One question is one task ([`convert_mmlu.py`](convert_mmlu.py)); name
order is stream order. The prompt is upstream's question block
(`Answer this multiple-choice question...`) under a preamble that names
the persistent workspace. The agent writes `{"answer": "<letter>"}` to
`/app/answer.json`, and the verifier
([`score_mmlu.py`](score_mmlu.py)) scores the letter against a key that
lives only in `tests/`, which the agent's container never mounts.
Parsing is upstream's: the JSON object, its near-misses, or a file
holding just the letter. A submission that names no single option scores
0 with a reason, so spraying every letter earns nothing.

Deviations from upstream, all forced by one task = one container:

- **Feedback arrives in the next instruction.** Upstream tells the model
  the correct option conversationally after each question; here that
  block opens the next task, quoting the question it refers to. It names
  the option unconditionally rather than saying "correct" or "incorrect",
  because the agent's own answer is in its memory, not in ours.
- **Memory is a directory, not a context window.** What survives between
  questions is what the agent writes to `$REEF_EVAL_STATE_DIR`.
- **Tasks run offline.** The questions are public MMLU-Pro items, so
  egress would measure retrieval rather than what the stream taught.

## The pin

`mmlu_pro/test_300.json` is content-pinned: `fetch.py` checks its sha256
and stops if upstream re-cuts the file, since that is a different
benchmark rather than a newer one. The dataset publishes no tags;
`--revision <sha>` pins the commit too. Conversion is deterministic, so
the same source file always produces the same 300 tasks.
