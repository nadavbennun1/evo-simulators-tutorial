#!/usr/bin/env python3
"""Start and stop a managed local preview without leaving an occupied port."""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SITE = Path(__file__).resolve().parents[1]


class PreviewServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def state_path(port: int) -> Path:
    return Path("/tmp") / f"drift-design-workshop-{os.getuid()}-{port}.json"


def read_state(port: int) -> dict | None:
    path = state_path(port)
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def process_is_managed(state: dict | None) -> bool:
    if not state or not isinstance(state.get("pid"), int):
        return False
    try:
        command = (Path("/proc") / str(state["pid"]) / "cmdline").read_bytes().replace(b"\0", b" ").decode()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return False
    return str(Path(__file__).resolve()) in command and " serve " in f" {command} "


def port_is_free(host: str, port: int) -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        probe.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        probe.close()


def serve(host: str, port: int) -> int:
    path = state_path(port)
    handler = partial(SimpleHTTPRequestHandler, directory=str(SITE))
    try:
        server = PreviewServer((host, port), handler)
    except OSError as error:
        print(f"Cannot preview on {host}:{port}: {error}", file=sys.stderr)
        print(f"Check with: {Path(__file__).name} status --port {port}", file=sys.stderr)
        return 1
    own_state = {"pid": os.getpid(), "host": host, "port": port, "site": str(SITE)}
    path.write_text(json.dumps(own_state) + "\n")
    stopping = False

    def request_stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    server.timeout = 0.4
    print(f"Serving {SITE} at http://{host}:{port}/ (PID {os.getpid()})", flush=True)
    try:
        while not stopping:
            server.handle_request()
    finally:
        server.server_close()
        if read_state(port) == own_state:
            path.unlink(missing_ok=True)
        print(f"Preview stopped; {host}:{port} released", flush=True)
    return 0


def start(host: str, port: int) -> int:
    state = read_state(port)
    if process_is_managed(state):
        print(f"Preview already running at http://{host}:{port}/ (PID {state['pid']})")
        return 0
    state_path(port).unlink(missing_ok=True)
    if not port_is_free(host, port):
        print(f"Port {port} is occupied by an unmanaged process.", file=sys.stderr)
        print(f"Stop that process or use: {Path(__file__).name} start --port {port + 1}", file=sys.stderr)
        return 1
    log_path = Path("/tmp") / f"drift-design-workshop-{os.getuid()}-{port}.log"
    log = log_path.open("ab")
    subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "serve", "--host", host, "--port", str(port)],
        cwd=SITE, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
    )
    log.close()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        state = read_state(port)
        if process_is_managed(state):
            print(f"Preview started at http://{host}:{port}/ (PID {state['pid']})")
            print(f"Stop it with: python workshop_site/scripts/preview.py stop --port {port}")
            return 0
        time.sleep(.05)
    print(f"Preview failed to start; inspect {log_path}", file=sys.stderr)
    return 1


def stop(host: str, port: int) -> int:
    state = read_state(port)
    if not process_is_managed(state):
        state_path(port).unlink(missing_ok=True)
        if port_is_free(host, port):
            print(f"No managed preview is running; port {port} is free")
            return 0
        print(f"Port {port} is occupied, but not by this managed preview; it was not killed.", file=sys.stderr)
        return 1
    os.kill(state["pid"], signal.SIGTERM)
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        if not process_is_managed(state) and port_is_free(host, port):
            state_path(port).unlink(missing_ok=True)
            print(f"Preview PID {state['pid']} stopped; port {port} is free")
            return 0
        time.sleep(.1)
    print(f"Preview PID {state['pid']} did not release port {port} in time", file=sys.stderr)
    return 1


def status(host: str, port: int) -> int:
    state = read_state(port)
    if process_is_managed(state):
        print(f"Managed preview is running at http://{host}:{port}/ (PID {state['pid']})")
        return 0
    state_path(port).unlink(missing_ok=True)
    if port_is_free(host, port):
        print(f"Port {port} is free")
        return 0
    print(f"Port {port} is occupied by an unmanaged process")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("start", "stop", "status", "serve"), nargs="?", default="start")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    return {"start": start, "stop": stop, "status": status, "serve": serve}[args.command](args.host, args.port)


if __name__ == "__main__":
    raise SystemExit(main())
