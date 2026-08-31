# Benchmarks

Everything tide ships, in one table. Each benchmark is a directory of stock
Harbor tasks, so every task runs two ways:

```bash
tide run cl-bench/bsm-s01 --agent oracle                        # through tide
harbor trial start -p tasks/continual-learning/cl-bench/bsm-s01 # stock Harbor
```

`tide list` shows what is runnable where you are.

| Benchmark | Regime | Tasks | Run |
|---|---|---|---|
| [first-party](https://github.com/Human-Agent-Society/reef-eval/tree/main/tasks/autoresearch/first-party) | autoresearch | 6 | `tide run autoresearch/first-party --agent oracle` |
| [EdgeBench](https://github.com/Human-Agent-Society/reef-eval/tree/main/tasks/autoresearch/edgebench) | autoresearch | 51 | `tide run edgebench/<task> --budget <h>` |
| [FrontierCS](https://github.com/Human-Agent-Society/reef-eval/tree/main/tasks/autoresearch/frontier-cs) | autoresearch | 208 | `tide run frontier-cs/<task> --agent <a>` |
| [terminal-bench](https://github.com/Human-Agent-Society/reef-eval/tree/main/tasks/continual-learning/terminal-bench) | stream | 89 | `tide stream terminal-bench --agent <a>` |
| [CL-Bench](https://github.com/Human-Agent-Society/reef-eval/tree/main/tasks/continual-learning/cl-bench) | stream | 301 | `tide stream cl-bench --agent <a>` |
| [SWE-bench Verified](https://github.com/Human-Agent-Society/reef-eval/tree/main/tasks/continual-learning/swebench-verified) | stream | 500 | `tide fetch swebench-verified --limit 50` first |

The first-party tasks each teach one hard part of autoresearch, and each
benchmark states its upstream, license, and oracle scores in
[the catalog](https://github.com/Human-Agent-Society/reef-eval/tree/main/tasks),
next to the tasks themselves.

To write your own, start from
[`tasks/_template`](https://github.com/Human-Agent-Society/reef-eval/tree/main/tasks/_template)
and follow [Authoring tasks](authoring-tasks.md).
