"""IDE-friendly launcher for the static Evolution × Inference workshop."""
from __future__ import annotations

from workshop_site.scripts.preview import serve

SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8765


def main() -> int:
    print(
        f"Evolution × Inference workshop is starting on its fixed forwarded port "
        f"{SERVER_PORT}. Use the IDE's Open in Browser notification or Ports panel.",
        flush=True,
    )
    return serve(SERVER_HOST, SERVER_PORT)


if __name__ == "__main__":
    raise SystemExit(main())
