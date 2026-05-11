"""Stdout transport — default. Prints to the terminal."""
from __future__ import annotations

import sys
from datetime import datetime, timezone


class StdoutTransport:
    def __init__(self, name: str = "agent"):
        self.name = name

    def post(self, text: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        print(f"\n─── {self.name} @ {ts}Z ───", flush=True)
        print(text, flush=True)
        print("─" * 40, flush=True)
        sys.stdout.flush()
