"""BrowseComp+ judge: the corpus, the answer and the tool budget for one query.

Runs in the judge container with a read-only FTS index of the corpus and
the short answer the query is scored against; the agent reaches both
only through these endpoints. Semantics are AgentCL's
``browsecomp_runner``: search returns the top 5, a document opens to
12,000 characters, a hundred tool calls per query, and the answer is
graded by upstream's normalised exact match -- NFKC, casefolded,
punctuation collapsed. Answering is terminal.

Endpoints: ``POST /search`` {"query"} — ``POST /open`` {"doc_id"} —
``POST /answer`` {"answer"}, terminal — ``GET /final``, the verifier's
verdict — ``GET /health``.
"""

import json
import os
import re
import threading
import unicodedata
from pathlib import Path

from browsecomp_corpus import BrowseCompCorpus
from judge_http import port_from_env, serve

JUDGE_DIR = Path(os.environ.get("JUDGE_DIR", Path(__file__).parent))
INSTANCE = json.loads((JUDGE_DIR / "instance.json").read_text())
CORPUS_PATH = os.environ.get("CORPUS_PATH", str(JUDGE_DIR / "corpus.sqlite"))

#: Upstream's defaults: --max-tool-calls, --search-top-k, --open-max-chars.
MAX_TOOL_CALLS = 100
SEARCH_TOP_K = 5
OPEN_MAX_CHARS = 12_000

_lock = threading.Lock()
_state = {"calls": 0, "done": False, "correct": False, "submitted": None}
_corpus = None


def corpus() -> BrowseCompCorpus:
    global _corpus
    if _corpus is None:
        _corpus = BrowseCompCorpus(CORPUS_PATH)
    return _corpus


def normalize_answer(value: str) -> str:
    """Upstream's normalisation, ported: NFKC, casefold, punctuation out."""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def exact_match(prediction: str, reference: str) -> bool:
    return bool(normalize_answer(reference)) and normalize_answer(
        prediction
    ) == normalize_answer(reference)


def _spend_call() -> dict | None:
    """Count one tool call; returns a refusal once the budget is gone."""
    if _state["calls"] >= MAX_TOOL_CALLS:
        return {"error": "tool budget exhausted", "calls_used": _state["calls"]}
    _state["calls"] += 1
    return None


def search(body: dict) -> tuple[dict, int]:
    query = str(body.get("query", "")).strip()
    if not query:
        return {"error": "send {'query': '<words>'}"}, 400
    with _lock:
        if _state["done"]:
            return {"error": "query is finished"}, 409
        refusal = _spend_call()
        if refusal:
            return refusal, 429
        results = corpus().search(query, limit=SEARCH_TOP_K)
        return {
            "results": results,
            "calls_used": _state["calls"],
            "calls_remaining": MAX_TOOL_CALLS - _state["calls"],
        }, 200


def open_document(body: dict) -> tuple[dict, int]:
    doc_id = str(body.get("doc_id", "")).strip()
    if not doc_id:
        return {"error": "send {'doc_id': '<id from search>'}"}, 400
    with _lock:
        if _state["done"]:
            return {"error": "query is finished"}, 409
        refusal = _spend_call()
        if refusal:
            return refusal, 429
        document = corpus().open(doc_id, max_chars=OPEN_MAX_CHARS)
        if document is None:
            return {
                "error": f"no document {doc_id!r}",
                "calls_used": _state["calls"],
                "calls_remaining": MAX_TOOL_CALLS - _state["calls"],
            }, 404
        return {
            **document,
            "calls_used": _state["calls"],
            "calls_remaining": MAX_TOOL_CALLS - _state["calls"],
        }, 200


def answer(body: dict) -> tuple[dict, int]:
    submitted = str(body.get("answer", ""))
    with _lock:
        if _state["done"]:
            return {"error": "query is finished", **verdict()}, 409
        # Answering is free and terminal, as upstream: one shot, no probing.
        _state["submitted"] = submitted
        _state["correct"] = exact_match(submitted, INSTANCE["answer"])
        _state["done"] = True
        return verdict(), 200


def verdict() -> dict:
    return {
        "reward": 1.0 if _state["correct"] else 0.0,
        "correct": _state["correct"],
        "calls_used": _state["calls"],
        "query_id": INSTANCE["query_id"],
    }


def final(_body: dict) -> dict:
    with _lock:
        result = verdict()
        if _state["submitted"] is None:
            result["note"] = "no answer submitted"
        return result


if __name__ == "__main__":
    corpus()  # fail at startup if the corpus is not mounted
    serve(
        {
            ("POST", "/search"): search,
            ("POST", "/open"): open_document,
            ("POST", "/answer"): answer,
            ("GET", "/final"): final,
        },
        port_from_env(),
    )
