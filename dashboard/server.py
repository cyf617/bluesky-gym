"""Stdlib HTTP server for the training dashboard.

    python -m dashboard                                   # http://127.0.0.1:8050
    python -m dashboard --results competitor/results --port 8050
    python -m dashboard --host 0.0.0.0                    # expose on the network

Endpoints: ``/api/runs``, ``/api/run/<name>``, ``/api/run/<name>/replays``,
``/api/run/<name>/replay/ep<i>.json``, ``/api/tunables``; everything else is
served from ``dashboard/static``.
"""

from __future__ import annotations

import argparse
import json
import math
import mimetypes
import os
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

from dashboard import data

REPLAY_FILE = re.compile(r"^ep\d+\.json$")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
DEFAULT_RESULTS = os.path.join("competitor", "results")


def tunables_payload():
    from competitor import config
    return {
        "defaults": config.resolve_mdp_config(),
        "keys": list(config.TUNABLE_KEYS),
        "action_modes": list(config.ACTION_MODES),
    }


def _finite(obj):
    """NaN/inf -> None so browsers can JSON.parse the payload."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    return obj


def make_handler(results_dir):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send(self, status, body, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload, status=HTTPStatus.OK):
            body = json.dumps(_finite(payload), allow_nan=False, default=str).encode()
            self._send(status, body, "application/json; charset=utf-8")

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            try:
                if path == "/api/runs":
                    names = data.discover_runs(results_dir)
                    return self._json({
                        "results_dir": os.path.abspath(results_dir),
                        "runs": [data.run_summary(results_dir, n) for n in names],
                        "baselines": data.load_baselines(results_dir),
                    })
                if path.startswith("/api/run/"):
                    name, _, rest = path[len("/api/run/"):].partition("/")
                    if name not in data.discover_runs(results_dir):
                        return self._json({"error": f"unknown run {name!r}"}, HTTPStatus.NOT_FOUND)
                    if rest == "":
                        return self._json(data.run_detail(results_dir, name))
                    if rest == "replays":
                        return self._json(data.replay_index(results_dir, name))
                    if rest.startswith("replay/") and REPLAY_FILE.match(rest[len("replay/"):]):
                        file_path = os.path.join(results_dir, name, "replays", rest[len("replay/"):])
                        if os.path.isfile(file_path):
                            with open(file_path, "rb") as f:
                                return self._send(HTTPStatus.OK, f.read(), "application/json; charset=utf-8")
                    return self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                if path == "/api/tunables":
                    return self._json(tunables_payload())
            except Exception as exc:  # surface parse errors in the UI instead of a dead socket
                return self._json({"error": f"{type(exc).__name__}: {exc}"},
                                  HTTPStatus.INTERNAL_SERVER_ERROR)
            self._static(path)

        def _static(self, path):
            rel = "index.html" if path in ("", "/") else path.lstrip("/")
            full = os.path.realpath(os.path.join(STATIC_DIR, rel))
            if not full.startswith(os.path.realpath(STATIC_DIR) + os.sep) or not os.path.isfile(full):
                return self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
            ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
            if ctype.startswith("text/") or ctype.endswith("javascript"):
                ctype += "; charset=utf-8"
            with open(full, "rb") as f:
                self._send(HTTPStatus.OK, f.read(), ctype)

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=DEFAULT_RESULTS,
                        help="directory that holds one sub-directory per run")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args(argv)

    server = ThreadingHTTPServer((args.host, args.port), make_handler(args.results))
    print(f"dashboard: http://{args.host}:{args.port}  (results: {os.path.abspath(args.results)})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
