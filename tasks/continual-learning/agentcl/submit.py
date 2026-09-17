#!/usr/bin/env python3
"""Submit a completion to the judge: ``submit /app/completion.py``.

Lives in the agent's container because the payload is a file of Python,
which is awkward to quote into curl. Prints the judge's verdict: whether
the hidden tests passed, what they said, and how many submissions are
left.
"""

import json
import os
import sys
import urllib.error
import urllib.request


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: submit <file.py>", file=sys.stderr)
        return 2
    path = sys.argv[1]
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            completion = handle.read()
    except OSError as error:
        print(f"cannot read {path}: {error}", file=sys.stderr)
        return 2
    url = os.environ.get("JUDGE_URL", "http://judge:8082") + "/submit"
    request = urllib.request.Request(
        url,
        data=json.dumps({"completion": completion}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        payload = json.load(error)
    except urllib.error.URLError as error:
        print(f"judge unreachable at {url}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=1))
    return 0 if payload.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
