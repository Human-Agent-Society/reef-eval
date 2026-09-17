"""Fetch AgentCL's MMLU-Pro stream and convert it to Harbor tasks.

    cd tasks/continual-learning/agentcl
    python fetch.py                        # all three domains, 300 tasks
    python fetch.py economics              # one stream
    python fetch.py economics --limit 10   # its first N questions

AgentCL is CC-BY-NC-4.0 and this repo is Apache-2.0, so no converted
task is committed here; this script builds them on your machine, the way
SWE-bench Verified's fetch.py does for its own licensing reason.

The download is content-pinned rather than commit-pinned: the source
file must hash to ``DIGEST`` below, so an upstream re-cut stops the run
instead of quietly changing what the benchmark measures. Pass
``--revision <sha>`` to pin the dataset commit as well.
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

REPO = "osunlp/AgentCL"
# The dataset publishes no tags, so "main" plus the digest below is the pin.
DEFAULT_REVISION = "main"
SOURCE = "mmlu_pro/test_300.json"
DIGEST = "bcfca99dece36ad04b8f113f8feb75a672b096275889194a5a039b4a4a93b2ee"

#: Upstream streams 100 questions per domain. A short domain means the
#: source moved under us: a different benchmark, not a smaller one.
QUESTIONS_PER_DOMAIN = 100


def _module(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def download(revision: str) -> Path:
    """The source file, cached under ``.data`` after the first fetch."""
    path = CACHE / SOURCE.replace("/", "__")
    if path.exists():
        return path
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{SOURCE}"
    CACHE.mkdir(parents=True, exist_ok=True)
    print(f"downloading {SOURCE} @ {revision} ...", flush=True)
    try:
        urllib.request.urlretrieve(url, path)
    except (urllib.error.URLError, OSError) as error:
        path.unlink(missing_ok=True)
        raise SystemExit(f"could not download {url}: {error}") from error
    return path


def verify(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != DIGEST:
        raise SystemExit(
            f"{SOURCE} integrity check failed: sha256 {digest[:16]}... does not "
            f"match the declared {DIGEST[:16]}.... Upstream may have re-cut the "
            f"file; delete {CACHE} and re-run with --revision <sha> to pin the "
            "commit these tasks were built from."
        )


def main() -> None:
    convert = _module("convert_mmlu")
    parser = argparse.ArgumentParser(
        description="fetch AgentCL's MMLU-Pro stream and convert it (pinned)"
    )
    parser.add_argument(
        "domains",
        nargs="*",
        metavar="DOMAIN",
        help=f"domains to convert (default: all of {', '.join(convert.DOMAINS)})",
    )
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N", help="first N per domain"
    )
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help=f"dataset revision to download (default: {DEFAULT_REVISION})",
    )
    args = parser.parse_args()

    domains = args.domains or list(convert.DOMAINS)
    unknown = [domain for domain in domains if domain not in convert.DOMAINS]
    if unknown:
        raise SystemExit(
            f"unknown domain(s) {', '.join(unknown)}; "
            f"AgentCL cuts MMLU-Pro into {', '.join(convert.DOMAINS)}"
        )

    source = download(args.revision)
    verify(source)
    rows = json.loads(source.read_text())

    written = 0
    for domain in domains:
        found = len(convert.load_domain(rows, domain))
        if args.limit is None and found != QUESTIONS_PER_DOMAIN:
            raise SystemExit(
                f"expected {QUESTIONS_PER_DOMAIN} {domain} questions, found {found}"
            )
        tasks = convert.convert_domain(rows, HERE, domain=domain, limit=args.limit)
        written += len(tasks)
        print(f"{domain}: {len(tasks)} task(s)")

    print(f"wrote {written} AgentCL MMLU-Pro task(s) -> {HERE}")
    print(
        "stream one domain: reef-eval stream "
        f"tasks/continual-learning/agentcl/mmlu-{domains[0]}-* --agent <a>"
    )


if __name__ == "__main__":
    main()
