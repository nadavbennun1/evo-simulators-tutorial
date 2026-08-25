from __future__ import annotations

import functools
import http.server
import threading
import urllib.request
from pathlib import Path

SITE = Path(__file__).resolve().parents[1]


def test_local_static_server_smoke():
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(SITE))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        for path in ("/index.html", "/evolution.html", "/sbi.html", "/data/zhou_manifest.json"):
            with urllib.request.urlopen(base + path, timeout=3) as response:
                assert response.status == 200 and response.read()
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=3)
    # The preview command that follows the tests must be able to reuse the port.
    rebound = http.server.ThreadingHTTPServer(("127.0.0.1", server.server_port), handler)
    rebound.server_close()
