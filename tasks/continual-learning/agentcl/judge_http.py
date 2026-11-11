"""Shared HTTP plumbing for the AgentCL judge sidecars.

Four of the five subsets keep something out of the agent's reach -- the
hidden tests, the corpus, the environment -- so each ships a judge
container the agent talks to over HTTP. They differ only in what the
endpoints mean, so the routing, the JSON handling and ``/health`` live
here and each server keeps its own semantics.

A handler takes the decoded request body (an empty dict on GET) and
returns either a payload or a ``(payload, status)`` pair. Anything it
raises becomes a 500 with the exception text, so a broken judge is
visible in the trial log rather than a hung agent.
"""

import json
import os
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

Route = Callable[[dict], dict | tuple[dict, int]]

MAX_BODY_BYTES = 8 * 1024 * 1024


def port_from_env(default: int = 8082) -> int:
    return int(os.environ.get("PORT", str(default)))


def serve(routes: dict[tuple[str, str], Route], port: int) -> None:
    """Serve ``{(method, path): handler}`` forever on *port*."""

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _dispatch(self, method: str, body: dict) -> None:
            path = self.path.split("?", 1)[0]
            handler = routes.get((method, path))
            if handler is None:
                return self._reply({"error": f"unknown endpoint {path}"}, 404)
            try:
                result = handler(body)
            except Exception as error:  # noqa: BLE001 - a judge never dies quietly
                return self._reply({"error": f"{type(error).__name__}: {error}"}, 500)
            if isinstance(result, tuple):
                payload, status = result
                return self._reply(payload, status)
            return self._reply(result)

        def do_GET(self):
            if self.path.split("?", 1)[0] == "/health":
                return self._reply({"ok": True})
            self._dispatch("GET", {})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            if length > MAX_BODY_BYTES:
                return self._reply({"error": "request body too large"}, 413)
            raw = self.rfile.read(length)
            try:
                body = json.loads(raw) if raw else {}
            except (json.JSONDecodeError, ValueError):
                return self._reply({"error": "body must be JSON"}, 400)
            if not isinstance(body, dict):
                return self._reply({"error": "body must be a JSON object"}, 400)
            self._dispatch("POST", body)

        def log_message(self, *args):
            pass

    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
