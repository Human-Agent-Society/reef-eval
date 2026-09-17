"""Fetch AgentCL and convert its five subsets to Harbor tasks.

    cd tasks/continual-learning/agentcl
    python fetch.py                                  # everything but BrowseComp+
    python fetch.py mmlu codeeval                    # named subsets
    python fetch.py scienceworld --limit 10          # first N per stream
    python fetch.py browsecomp --corpus corpus.jsonl # needs the corpus
    python fetch.py codeeval --check-references      # re-measure, convert nothing

AgentCL is CC-BY-NC-4.0 and this repo is Apache-2.0, so no converted
task is committed here; this script builds them on your machine, the way
SWE-bench Verified's fetch.py does for its own licensing reason.

Every download is content-pinned: each source file must hash to its
entry in ``DIGESTS``, so an upstream re-cut stops the run instead of
quietly changing what the benchmark measures. Pass ``--revision <sha>``
to pin the dataset commit as well.

BrowseComp+ needs its corpus, which is published apart from AgentCL and
is not redistributable here: pass ``--corpus`` a JSONL or TSV of
``doc_id``/``title``/``text``. It is indexed once, under ``.data``, and
mounted read-only into every BrowseComp+ judge.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from types import ModuleType

HERE = Path(__file__).parent
CACHE = HERE / ".data"
CORPUS_INDEX = CACHE / "browsecomp_corpus.sqlite"

REPO = "osunlp/AgentCL"
# The dataset publishes no tags, so "main" plus the digests below are the pin.
DEFAULT_REVISION = "main"

DIGESTS = {
    "mmlu_pro/test_300.json": "bcfca99dece36ad04b8f113f8feb75a672b096275889194a5a039b4a4a93b2ee",
    "codeeval-pro/humaneval_pro.naive.json": "91b23b2a812c3b8d8603293b234bf80a199fbdcd43306af1a6d59f9d667022b2",
    "codeeval-pro/humaneval_pro.compositional.json": "2cb5a5214970f256c54aa5ceaf73a41a1f8915565c27735f46cfdf4662af6d9c",
    "codeeval-pro/mbpp_pro.naive.json": "9190061a61aaa12b0ec5efc2152325fcd61345c03eefca1d696752fdd09dbfdc",
    "codeeval-pro/mbpp_pro.compositional.json": "9d01d7d90a6af7691e47b70ef8b7d5deb60bb3fbbc50f53b1bc628c0a46194d2",
    "codeeval-pro/bigcodebench_lite_pro.naive.json": "55d18f52f7e2562514dfc5bb4d75b161ce58861b1e50ea298e4e0033bbe0b636",
    "codeeval-pro/bigcodebench_lite_pro.compositional.json": "cd462894801f087866019d12d85f669d1ddd72b97ef722dcd180587868ad6df2",
    "browsecomp_plus/ground_truth_original.jsonl": "658bf946b84f44a983cef1ab3923d9d22a78f624e2c5288d4395152f6a65d1e2",
    "browsecomp_plus/ground_truth_subqueries.jsonl": "3f75c607bae98fa6756e556d9593868b439a13c1da09229485003ad3304e9ef4",
    "browsecomp_plus/qrel_golds.txt": "df3158916afb1909fb0b177b02253c6ea381a94095262c818b1e7a6f493993cf",
    "browsecomp_plus/qrel_evidence.txt": "3177494e64e6d8fe1b5d09cbc3d62a1230d1ea4275e101097164d00389966ec1",
    "agentboard_babyai/naive.jsonl": "dc923a597113b8aa569091b870d46d537abc5c3d07b8cb34486a20c763369e10",
    "agentboard_babyai/compositional.jsonl": "37f4dc7c6c98570de5d6d0cb37e179b49196a28e50d63c23839daddf9a5a7010",
    "agentboard_babyai/seeds.json": "bfb170be936c3ce1b2f0bd2e4f8bad128e234e122cdb55c50ee3314ae3343a8b",
    "agentboard_scienceworld/naive.jsonl": "ed58212de963cd3c3479e87a25c5d91e2d2decd4bbc3a39bb75283892d186ce1",
    "agentboard_scienceworld/block.jsonl": "fc7ad7c614782b7d1ec6fcb409149a66caa864f030500a80a0cdd57626d3b269",
}

#: Upstream's stream lengths. A stream that arrives shorter is a moved
#: source, not a smaller benchmark, so conversion stops.
EXPECTED = {
    "mmlu": 100,  # per domain
    "babyai": {"naive": 40, "comp": 56},
    "scienceworld": {"naive": 90, "block": 90},
    "browsecomp": {"naive": 100, "comp": 408},
}

SUBSETS = ("mmlu", "codeeval", "browsecomp", "babyai", "scienceworld")
#: BrowseComp+ is not in the default set: it cannot build without a corpus.
DEFAULT_SUBSETS = ("mmlu", "codeeval", "babyai", "scienceworld")


def _module(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def source(rel_path: str, revision: str) -> Path:
    """One pinned source file, downloaded on first use and hash-checked."""
    if rel_path not in DIGESTS:
        raise SystemExit(f"{rel_path} has no declared digest")
    path = CACHE / rel_path.replace("/", "__")
    if not path.exists():
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{rel_path}"
        CACHE.mkdir(parents=True, exist_ok=True)
        print(f"downloading {rel_path} @ {revision} ...", flush=True)
        try:
            urllib.request.urlretrieve(url, path)
        except (urllib.error.URLError, OSError) as error:
            path.unlink(missing_ok=True)
            raise SystemExit(f"could not download {url}: {error}") from error
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != DIGESTS[rel_path]:
        raise SystemExit(
            f"{rel_path} integrity check failed: sha256 {digest[:16]}... does not "
            f"match the declared {DIGESTS[rel_path][:16]}.... Upstream may have "
            f"re-cut the file; delete {CACHE} and re-run with --revision <sha> to "
            "pin the commit these tasks were built from."
        )
    return path


# ------------------------------------------------------------------ subsets


def fetch_mmlu(args) -> int:
    """MMLU-Pro: three 100-question domain streams."""
    convert = _module("convert_mmlu")
    rows = json.loads(source("mmlu_pro/test_300.json", args.revision).read_text())
    written = 0
    for domain in convert.DOMAINS:
        found = len(convert.load_domain(rows, domain))
        if args.limit is None and found != EXPECTED["mmlu"]:
            raise SystemExit(
                f"expected {EXPECTED['mmlu']} {domain} questions, found {found}"
            )
        tasks = convert.convert_domain(rows, HERE, domain=domain, limit=args.limit)
        written += len(tasks)
        print(f"  mmlu {domain}: {len(tasks)} task(s)")
    return written


def _codeeval_rows(convert, dataset: str, stream: str, revision: str) -> list[dict]:
    stem, suffix = convert.DATASETS[dataset], convert.STREAMS[stream]
    return convert.load_stream(source(f"codeeval-pro/{stem}.{suffix}.json", revision))


def fetch_codeeval(args) -> int:
    """CodeEval-Pro: three sources, each streamed naive and compositional."""
    convert = _module("convert_codeeval")
    written = 0
    for dataset in convert.DATASETS:
        for stream in convert.STREAMS:
            rows = _codeeval_rows(convert, dataset, stream, args.revision)
            tasks = convert.convert_stream(
                rows, HERE, dataset=dataset, stream=stream, limit=args.limit
            )
            written += len(tasks)
            print(f"  codeeval {dataset} {stream}: {len(tasks)} task(s)")
    broken = len(convert.REFERENCE_FAILURES)
    print(
        f"  codeeval: {broken} problems ship a reference that cannot pass their "
        "own tests (oracle_score = 0.0; see README)"
    )
    return written


def check_codeeval_references(args) -> int:
    """Re-run every published reference and report drift from the table.

    The judge's own program builder and runner do the work, so this
    measures exactly what a trial would. Problems appear in both streams;
    each is checked once. BigCodeBench needs the judge image's packages,
    so a missing module here is a gap in this environment, not a defect
    in the data -- run it inside the judge image to cover that dataset.
    """
    convert = _module("convert_codeeval")
    failures: dict[tuple[str, str, str], str] = {}
    seen: set[tuple[str, str, str]] = set()
    for dataset in convert.DATASETS:
        for stream in convert.STREAMS:
            for row in _codeeval_rows(convert, dataset, stream, args.revision):
                key = (dataset, row["category"], str(row["id"]))
                if key in seen:
                    continue
                seen.add(key)
                prefix, reference = convert.split_prefix(row)
                passed, feedback = _run_reference(prefix, row["test_code"], reference)
                if not passed:
                    failures[key] = feedback.strip().splitlines()[-1][:110]

    unrunnable = {k: v for k, v in failures.items() if "ModuleNotFoundError" in v}
    defects = {k: v for k, v in failures.items() if k not in unrunnable}
    new = {k: v for k, v in defects.items() if k not in convert.REFERENCE_FAILURES}
    gone = [
        key
        for key in convert.REFERENCE_FAILURES
        if key not in defects and key not in unrunnable
    ]
    print(f"checked {len(seen)} problems; {len(defects)} references cannot pass")
    for key, reason in sorted(defects.items()):
        print(f"  {key}: {reason}{'  (not in the table)' if key in new else ''}")
    if unrunnable:
        print(
            f"{len(unrunnable)} problems could not run here for want of packages; "
            "check those inside the judge image"
        )
    if gone:
        print(f"the table lists {len(gone)} problems that now pass: {gone}")
    if new or gone:
        raise SystemExit("REFERENCE_FAILURES is stale; update convert_codeeval.py")
    return 0


def _run_reference(prefix: str, test_code: str, completion: str) -> tuple[bool, str]:
    """Run one published reference through the judge's own harness."""
    import os
    import tempfile

    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    with tempfile.TemporaryDirectory() as workdir:
        (Path(workdir) / "instance.json").write_text(
            json.dumps(
                {"problem_id": "probe", "prefix": prefix, "test_code": test_code}
            )
        )
        os.environ["JUDGE_DIR"] = workdir
        sys.modules.pop("codeeval_server", None)
        return _module("codeeval_server").run_tests(completion)


def fetch_babyai(args) -> int:
    """AgentBoard BabyAI: the naive and compositional episode streams."""
    convert = _module("convert_babyai")
    seeds = convert.load_seeds(source("agentboard_babyai/seeds.json", args.revision))
    written = 0
    for stream, released in convert.STREAMS.items():
        rows = convert.load_stream(
            source(f"agentboard_babyai/{released}.jsonl", args.revision)
        )
        if args.limit is None and len(rows) != EXPECTED["babyai"][stream]:
            raise SystemExit(
                f"expected {EXPECTED['babyai'][stream]} babyai {stream} episodes, "
                f"found {len(rows)}"
            )
        tasks = convert.convert_stream(
            rows, seeds, HERE, stream=stream, limit=args.limit
        )
        written += len(tasks)
        print(f"  babyai {stream}: {len(tasks)} task(s)")
    return written


def fetch_scienceworld(args) -> int:
    """AgentBoard ScienceWorld: the naive and block orderings."""
    convert = _module("convert_scienceworld")
    written = 0
    for stream, released in convert.STREAMS.items():
        rows = convert.load_stream(
            source(f"agentboard_scienceworld/{released}.jsonl", args.revision)
        )
        if args.limit is None and len(rows) != EXPECTED["scienceworld"][stream]:
            raise SystemExit(
                f"expected {EXPECTED['scienceworld'][stream]} scienceworld "
                f"{stream} episodes, found {len(rows)}"
            )
        tasks = convert.convert_stream(rows, HERE, stream=stream, limit=args.limit)
        written += len(tasks)
        print(f"  scienceworld {stream}: {len(tasks)} task(s)")
    return written


def build_corpus_index(corpus_path: str, qrels: list[Path]) -> int:
    """Index the corpus once and check it holds every judged document."""
    corpus_module = _module("browsecomp_corpus")
    if not CORPUS_INDEX.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        print(f"indexing {corpus_path} ...", flush=True)
        corpus_module.build_index(
            corpus_module.iter_documents(corpus_path), CORPUS_INDEX
        )
    corpus = corpus_module.BrowseCompCorpus(CORPUS_INDEX)
    try:
        missing = corpus.missing_doc_ids(corpus_module.qrel_doc_ids(qrels))
        if missing:
            raise SystemExit(
                f"the corpus misses {len(missing)} judged documents "
                f"(first: {missing[:5]}); it is not the BrowseComp+ corpus"
            )
        return corpus.count()
    finally:
        corpus.close()


def fetch_browsecomp(args) -> int:
    """BrowseComp+: the originals, and the subqueries-then-originals stream."""
    convert = _module("convert_browsecomp")
    if not args.corpus and not CORPUS_INDEX.exists():
        raise SystemExit(
            "BrowseComp+ needs its corpus, which AgentCL does not ship: pass "
            "--corpus <jsonl|tsv of doc_id/title/text>. It is indexed once."
        )
    for name in (
        "browsecomp_plus/ground_truth_original.jsonl",
        "browsecomp_plus/ground_truth_subqueries.jsonl",
        "browsecomp_plus/qrel_golds.txt",
        "browsecomp_plus/qrel_evidence.txt",
    ):
        source(name, args.revision)
    data_dir = CACHE / "browsecomp_plus"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in ("ground_truth_original.jsonl", "ground_truth_subqueries.jsonl"):
        target = data_dir / name
        if not target.exists():
            target.write_bytes((CACHE / f"browsecomp_plus__{name}").read_bytes())

    documents = build_corpus_index(
        args.corpus,
        [
            CACHE / "browsecomp_plus__qrel_golds.txt",
            CACHE / "browsecomp_plus__qrel_evidence.txt",
        ],
    )
    print(f"  browsecomp corpus: {documents} documents indexed at {CORPUS_INDEX}")

    written = 0
    for stream in convert.STREAMS:
        rows = convert.load_stream(data_dir, stream)
        if args.limit is None and len(rows) != EXPECTED["browsecomp"][stream]:
            raise SystemExit(
                f"expected {EXPECTED['browsecomp'][stream]} browsecomp {stream} "
                f"queries, found {len(rows)}"
            )
        tasks = convert.convert_stream(rows, HERE, stream=stream, limit=args.limit)
        written += len(tasks)
        print(f"  browsecomp {stream}: {len(tasks)} task(s)")
    print(
        f"  export {convert.CORPUS_ENV_VAR}={CORPUS_INDEX.resolve()}  "
        "# every BrowseComp+ judge mounts this"
    )
    return written


FETCHERS = {
    "mmlu": fetch_mmlu,
    "codeeval": fetch_codeeval,
    "browsecomp": fetch_browsecomp,
    "babyai": fetch_babyai,
    "scienceworld": fetch_scienceworld,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="fetch AgentCL and convert it to Harbor tasks (pinned)"
    )
    parser.add_argument(
        "subsets",
        nargs="*",
        metavar="SUBSET",
        help=f"subsets to convert (default: {', '.join(DEFAULT_SUBSETS)})",
    )
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N", help="first N per stream"
    )
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help=f"dataset revision to download (default: {DEFAULT_REVISION})",
    )
    parser.add_argument(
        "--corpus",
        default="",
        metavar="PATH",
        help="BrowseComp+ corpus (JSONL or TSV of doc_id/title/text)",
    )
    parser.add_argument(
        "--check-references",
        action="store_true",
        help="codeeval only: re-run every published reference, convert nothing",
    )
    args = parser.parse_args()

    subsets = args.subsets or list(DEFAULT_SUBSETS)
    unknown = [name for name in subsets if name not in FETCHERS]
    if unknown:
        raise SystemExit(
            f"unknown subset(s) {', '.join(unknown)}; AgentCL has {', '.join(SUBSETS)}"
        )
    if args.check_references:
        if subsets != ["codeeval"]:
            raise SystemExit("--check-references applies to the codeeval subset")
        raise SystemExit(check_codeeval_references(args))
    if args.corpus and "browsecomp" not in subsets:
        subsets.append("browsecomp")

    written = 0
    for name in subsets:
        written += FETCHERS[name](args)
    if "browsecomp" not in subsets:
        print("skipped browsecomp: it needs --corpus (see this folder's README)")
    print(f"wrote {written} AgentCL task(s) -> {HERE}")
    print(
        "stream one of them: reef-eval stream "
        "tasks/continual-learning/agentcl/mmlu-economics-* --agent <a>"
    )


if __name__ == "__main__":
    main()
