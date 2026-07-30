from __future__ import annotations

import argparse
import os
import sys

import httpx


COMMANDS = [
    {"command": "start", "description": "Show Friendfolio commands"},
    {"command": "help", "description": "Show Friendfolio commands"},
    {"command": "whoami", "description": "Show your Telegram user ID"},
    {"command": "add", "description": "Classify and propose a note"},
    {"command": "friend", "description": "Create or find a friend"},
    {"command": "project", "description": "Create or find a project"},
    {"command": "friends", "description": "List friends"},
    {"command": "projects", "description": "List projects"},
    {"command": "show", "description": "Show a friend or project"},
    {"command": "next", "description": "Show pending follow-ups"},
    {"command": "followups", "description": "Show pending follow-ups"},
    {"command": "done", "description": "Complete a follow-up"},
    {"command": "inbox", "description": "Show uncategorized notes"},
    {"command": "reclassify", "description": "Retry an inbox note"},
    {"command": "birthdays", "description": "Show saved birthdays"},
    {"command": "search", "description": "Search notes"},
    {"command": "confidence", "description": "View or set confidence threshold"},
    {"command": "migrate", "description": "Preview a friend-note schema migration"},
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
