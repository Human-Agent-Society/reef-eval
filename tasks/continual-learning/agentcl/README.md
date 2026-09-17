# AgentCL

[AgentCL](https://huggingface.co/datasets/osunlp/AgentCL) (OSU NLP,
CC-BY-NC-4.0): task streams for measuring whether an agent gets better at
a domain as it works through it. Their framework is an agent plus a
memory method run over a stream; here that is an agent plus the carried
`$REEF_EVAL_STATE_DIR`, and the gain their tables report is
`metrics.transfer` against the same tasks run in isolation.

The dataset is non-commercial and this repo is Apache-2.0, so **no
AgentCL task is committed here**. [`fetch.py`](fetch.py) builds them on
your machine, the way SWE-bench Verified's does for its own licensing
reason.

## The five subsets, all converted

Every subset ships two orderings of the same material: a plain one and
one built so the second half needs what the first half taught. The
difference between them is the experiment.

| Stream | Tasks | One task | Reward |
|---|---|---|---|
| `mmlu-<domain>-qNNN` | 3 × 100 | an MMLU-Pro question in economics, engineering or philosophy | 1.0 for the keyed option |
| `code-<dataset>-naive-NNN` | 120 / 368 / 48 | a composed HumanEval, MBPP or BigCodeBench-Lite problem | 1.0 if the hidden tests pass |
| `code-<dataset>-comp-NNN` | 240 / 736 / 96 | the same, with every raw problem first | 1.0 if the hidden tests pass |
| `browsecomp-naive-NNN` | 100 | a BrowseComp+ research question over an offline corpus | 1.0 for the exact answer |
| `browsecomp-comp-NNN` | 408 | the same, with the 308 subqueries first | 1.0 for the exact answer |
| `babyai-naive-NNN` / `-comp-NNN` | 40 / 56 | an AgentBoard BabyAI room | progress: the fraction of checkpoints reached |
| `sciworld-naive-NNN` / `-block-NNN` | 90 / 90 | an AgentBoard ScienceWorld task | progress: the fraction of checkpoints reached |

2,692 tasks in 15 streams. Progress is what AgentCL reports as PR; the
environments also report SR (the simulator's own completion) in the
verifier log beside it.

```bash
reef-eval fetch agentcl                    # every subset but BrowseComp+
reef-eval fetch agentcl mmlu --limit 10    # one subset, or a smoke-sized prefix
reef-eval stream tasks/continual-learning/agentcl/mmlu-economics-* --agent claude-code
```

Stream **one stream at a time**: `reef-eval stream agentcl` would run all
2,692 tasks as one concatenation, which is not an experiment anyone ran.
The isolated baseline for `metrics.transfer` is `reef-eval run` over the
same task list.

## What runs where

MMLU-Pro is a plain task: read the question, write
`{"answer": "<letter>"}`, and a verifier scores the letter against a key
that lives in `tests/`, which the agent's container never mounts.

The other four keep something out of the agent's reach, so each task
starts a judge sidecar the agent talks to over HTTP —
[`judge_http.py`](judge_http.py) is the shared plumbing,
[`sidecar.py`](sidecar.py) the shared Harbor wiring:

- [`codeeval_server.py`](codeeval_server.py) holds the hidden tests and
  meters submissions the way upstream meters retries: ten tries, each
  returning what the tests said, 1.0 if one of them passes.
- [`browsecomp_server.py`](browsecomp_server.py) holds the corpus index
  and the answer: search the top 5, open 12,000 characters, a hundred
  tool calls, and answering is terminal.
- [`babyai_server.py`](babyai_server.py) and
  [`scienceworld_server.py`](scienceworld_server.py) hold the simulator,
  the subgoals and the score; the agent sends actions and gets
  observations, 30 steps per episode.

Because the judge owns the score, an agent with root in its own
container still cannot reach the thing it is being graded against.

## Deviations from upstream

- **Feedback arrives in the next instruction.** Upstream tells the model
  the correct MMLU-Pro option conversationally; here that block opens the
  next task, quoting the question it refers to.
- **Memory is a directory, not a context window.** What survives between
  tasks is what the agent writes to `$REEF_EVAL_STATE_DIR`.
- **Everything runs offline**, judge aside. These are public benchmarks;
  egress would measure lookup rather than what the stream taught.
- **Interaction is HTTP, not a chat turn.** The action spaces, budgets
  and scoring rules are upstream's; only the transport differs.

## Two things the data decides for us

**Seven CodeEval-Pro problems cannot be solved.** Their own published
reference fails their own tests — five prompts annotate `List`/`Tuple`
without importing `typing`, one test redefines the entry point after the
candidate, one reference calls a helper with the wrong shape. Upstream
builds the test program the same way, so it scores them the same way;
they are converted as they are, and those tasks record
`oracle_score = 0.0` with the reason in `reference_note`. Re-measure with
`reef-eval fetch agentcl codeeval --check-references`, which re-runs
every reference and fails if the table in
[`convert_codeeval.py`](convert_codeeval.py) has drifted. BigCodeBench's
references need the judge image's packages, so run that check there.

**BabyAI and ScienceWorld have no reference solution.** No gold
trajectory ships with the episodes, and ScienceWorld's own gold sequence
cannot be handed to the solution container without handing it to the
agent on the same network. Their `solution/solve.sh` looks once and
stops: it proves the pipeline, not a ceiling, and `task.toml` says so in
`reference = "pipeline-check"`.

## The pins

The dataset publishes no tags, so `fetch.py` pins a commit
(`e85c86e1`) and, independently, the sha256 of every source file it
reads. Either check failing stops the run, since a re-cut source is a
different benchmark rather than a newer one. Conversion is
deterministic: the same sources always produce the same tasks, and
`--limit` yields a byte-identical prefix of the full conversion.

The judge images pin what they can. BabyAI's environment comes from
[AgentGym](https://github.com/WooooDyy/AgentGym) at the commit AgentCL's
own servers were built against. ScienceWorld's wrapper is not versioned
upstream and is installed unpinned; pin it in
[`convert_scienceworld.py`](convert_scienceworld.py) once a build is
proven on your machine.

## BrowseComp+ needs its corpus

BrowseComp-Plus publishes queries and qrels; its corpus is separate and
not redistributable here, so `fetch.py` asks for it:

```bash
reef-eval fetch agentcl browsecomp --corpus <corpus.jsonl>   # doc_id/title/text
export AGENTCL_BROWSECOMP_CORPUS=.../agentcl/.data/browsecomp_corpus.sqlite
```

The corpus is indexed once into SQLite FTS5
([`browsecomp_corpus.py`](browsecomp_corpus.py)), checked against the
qrels so a wrong corpus stops the conversion, and mounted read-only into
every BrowseComp+ judge. Without `--corpus`, `fetch.py` builds the other
four subsets and says it skipped this one.
