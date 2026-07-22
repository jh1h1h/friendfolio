from __future__ import annotations

import argparse
import os
import sys

import httpx


COMMANDS = [
    {"command": "add", "description": "Classify and propose a note"},
    {"command": "friends", "description": "List friends"},
    {"command": "projects", "description": "List projects"},
    {"command": "next", "description": "Show project next actions"},
    {"command": "followups", "description": "Show pending follow-ups"},
    {"command": "inbox", "description": "Show uncategorized notes"},
    {"command": "birthdays", "description": "Show saved birthdays"},
    {"command": "search", "description": "Search notes"},
    {"command": "help", "description": "Show all commands"},
]


def telegram_call(
    token: str, method: str, payload: dict[str, object]
) -> dict[str, object]:
    response = httpx.post(
        f"https://api.telegram.org/bot{token}/{method}", json=payload, timeout=20
    )
    response.raise_for_status()
    body = response.json()
    if not body.get("ok"):
        raise RuntimeError(f"Telegram {method} failed: {body}")
    return body


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Register Friendfolio's Firebase webhook"
    )
    parser.add_argument("--project-id", required=True, help="Firebase project ID")
    args = parser.parse_args()
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    secret = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "")
    if not token or not secret:
        sys.exit(
            "Set TELEGRAM_BOT_TOKEN and TELEGRAM_WEBHOOK_SECRET in your shell first."
        )
    url = (
        f"https://asia-southeast1-{args.project_id}.cloudfunctions.net/telegram_webhook"
    )
    telegram_call(
        token,
        "setWebhook",
        {
            "url": url,
            "secret_token": secret,
            "allowed_updates": ["message", "callback_query"],
            "drop_pending_updates": False,
        },
    )
    telegram_call(token, "setMyCommands", {"commands": COMMANDS})
    print(f"Webhook registered: {url}")


if __name__ == "__main__":
    main()
