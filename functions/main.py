from __future__ import annotations

import hmac
import logging
import os

from firebase_admin import firestore, initialize_app
from firebase_functions import https_fn, logger, options, scheduler_fn

from friendfolio.deepseek import DeepSeekClassifier
from friendfolio.handlers import BotHandlers
from friendfolio.store import FirestoreRegistry
from friendfolio.telegram_api import TelegramAPI


initialize_app()
REGION = options.SupportedRegion.ASIA_SOUTHEAST1


def _allowed_user_ids() -> frozenset[int]:
    raw = os.environ.get("TELEGRAM_ALLOWED_USER_IDS", "")
    try:
        values = frozenset(int(part.strip()) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise RuntimeError(
            "TELEGRAM_ALLOWED_USER_IDS must be comma-separated integers"
        ) from exc
    if not values:
        raise RuntimeError("TELEGRAM_ALLOWED_USER_IDS is empty")
    return values


def _birthday_days() -> tuple[int, ...]:
    raw = os.environ.get("BIRTHDAY_REMINDER_DAYS", "7,1,0")
    try:
        values = tuple(
            sorted({int(part.strip()) for part in raw.split(",")}, reverse=True)
        )
    except ValueError as exc:
        raise RuntimeError("BIRTHDAY_REMINDER_DAYS must contain integers") from exc
    if not values or any(day < 0 or day > 366 for day in values):
        raise RuntimeError("BIRTHDAY_REMINDER_DAYS must contain values from 0 to 366")
    return values


def _handlers(require_deepseek: bool = True) -> BotHandlers:
    timezone_name = os.environ.get("APP_TIMEZONE", "Asia/Singapore")
    telegram = TelegramAPI(os.environ["TELEGRAM_BOT_TOKEN"])
    registry = FirestoreRegistry(firestore.client(), timezone_name)
    classifier = DeepSeekClassifier(
        os.environ.get("DEEPSEEK_API_KEY", "unused" if not require_deepseek else ""),
        os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro"),
        timezone_name,
    )
    try:
        expiry = int(os.environ.get("PENDING_EXPIRY_HOURS", "24"))
    except ValueError as exc:
        raise RuntimeError("PENDING_EXPIRY_HOURS must be an integer") from exc
    return BotHandlers(
        telegram=telegram,
        registry=registry,
        classifier=classifier,
        allowed_user_ids=_allowed_user_ids(),
        timezone_name=timezone_name,
        pending_expiry_hours=expiry,
        birthday_reminder_days=_birthday_days(),
    )


@https_fn.on_request(
    region=REGION,
    timeout_sec=60,
    max_instances=3,
    secrets=["TELEGRAM_BOT_TOKEN", "TELEGRAM_WEBHOOK_SECRET", "DEEPSEEK_API_KEY"],
)
def telegram_webhook(req: https_fn.Request) -> https_fn.Response:
    if req.method != "POST":
        return https_fn.Response("method not allowed", status=405)
    expected = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "")
    provided = req.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
    if not expected or not hmac.compare_digest(provided, expected):
        return https_fn.Response("unauthorized", status=401)
    update = req.get_json(silent=True)
    if not isinstance(update, dict):
        return https_fn.Response("invalid update", status=400)
    try:
        _handlers().handle_update(update)
    except Exception as exc:
        logging.exception("Telegram update failed")
        logger.error("Telegram update failed", error=str(exc))
        # Acknowledge the update to prevent an endless Telegram retry loop.
        return https_fn.Response("error acknowledged", status=200)
    return https_fn.Response("ok", status=200)


@scheduler_fn.on_schedule(
    schedule="0 9 * * *",
    timezone="Asia/Singapore",
    region=REGION,
    retry_count=1,
    secrets=["TELEGRAM_BOT_TOKEN"],
)
def daily_reminders(event: scheduler_fn.ScheduledEvent) -> None:
    del event
    _handlers(require_deepseek=False).send_daily_reminders()
