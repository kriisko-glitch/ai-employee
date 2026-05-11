"""Generic POST webhook transport — for Slack-incoming-webhooks, n8n, Zapier, etc."""
from __future__ import annotations

import json
import os
import urllib.request


class WebhookTransport:
    def __init__(self, url_env: str = "GENERIC_WEBHOOK_URL", format: str = "json"):
        self.url = os.environ.get(url_env, "").strip()
        if not self.url:
            raise RuntimeError(
                f"Webhook transport configured but {url_env} is not set."
            )
        self.format = format  # json | text

    def post(self, text: str) -> None:
        if self.format == "text":
            data = text.encode("utf-8")
            headers = {"Content-Type": "text/plain; charset=utf-8"}
        else:
            data = json.dumps({"text": text}).encode("utf-8")
            headers = {"Content-Type": "application/json"}
        req = urllib.request.Request(self.url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp.read()
