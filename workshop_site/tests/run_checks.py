#!/usr/bin/env python3
"""Dependency-free local runner for the plain-assert workshop checks."""
from __future__ import annotations

import importlib
import traceback


def main() -> int:
    modules = [importlib.import_module("test_workshop"), importlib.import_module("test_static_server")]
    tests = [(f"{module.__name__}.{name}", getattr(module, name)) for module in modules for name in dir(module) if name.startswith("test_")]
    failures = []
    for name, test in tests:
        try:
            test()
            print(f"PASS {name}")
        except Exception:
            failures.append(name)
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"\n{len(tests)-len(failures)} passed, {len(failures)} failed")
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
