from __future__ import annotations

import html
import logging
from datetime import UTC
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .deepseek import DeepSeekClassifier
from .models import NoteProposal, ProposalItem
from .store import FirestoreRegistry, safe_id, utc_now
from .telegram_api import TelegramAPI


LOGGER = logging.getLogger(__name__)

HELP_TEXT = """
<b>Friendfolio commands</b>

<code>/add note</code> — classify a note and ask before saving
<code>/friend name</code> / <code>/project name</code> — create an entity
<code>/friends</code> / <code>/projects</code> — list the registry
<code>/show friend Alice</code> — show a friend or project
<code>/next</code> — open project next actions
<code>/followups</code> — pending follow-ups
<code>/done ID</code> — complete an item
<code>/inbox</code> — uncategorized notes
<code>/reclassify ID context</code> — retry an inbox note
<code>/birthdays</code> — saved birthdays
<code>/search words</code> — search note text
<code>/whoami</code> — show your Telegram user ID
""".strip()


def _short(value: str, limit: int = 220) -> str:
    clean = " ".join(value.split())
    return clean if len(clean) <= limit else clean[: limit - 1] + "…"


def _display_id(note_id: str) -> str:
    return note_id[:8]


def _local_datetime(value: Any, timezone_name: str) -> str:
    if not isinstance(value, datetime):
        return "no date"
    return value.astimezone(ZoneInfo(timezone_name)).strftime("%d %b %Y, %H:%M")


def _now_in_timezone(timezone_name: str) -> datetime:
    try:
        return datetime.now(ZoneInfo(timezone_name))
    except ZoneInfoNotFoundError:
        LOGGER.warning("Timezone %s not found; defaulting to UTC", timezone_name)
        return datetime.now(UTC)


def _proposal_preview(proposal: NoteProposal, timezone_name: str) -> str:
    lines = [f"<b>Proposed update</b> — {html.escape(proposal.summary)}", ""]
    for index, item in enumerate(proposal.items, 1):
        target = "Inbox" if item.target_type == "uncategorized" else item.target_name
        lines.append(
            f"<b>{index}. {html.escape(item.target_type.title())}: "
            f"{html.escape(target)}</b> · <code>{html.escape(item.category)}</code>"
        )
        lines.append(html.escape(_short(item.content, 500)))
        details = []
        if item.birthday_mm_dd:
            details.append(f"birthday {item.birthday_mm_dd}")
        if item.follow_up_at:
            local = item.follow_up_at
            if local.tzinfo is None:
                local = local.replace(tzinfo=ZoneInfo(timezone_name))
            details.append(
                f"follow up {local.astimezone(ZoneInfo(timezone_name)):%d %b %Y, %H:%M}"
            )
        details.append(f"confidence {item.confidence:.0%}")
        lines.append(" · ".join(details))
        lines.append(f"<i>{html.escape(item.reason)}</i>\n")
    lines.append(
        "Nothing has been added to the registry yet. Approve this exact proposal?"
    )
    text = "\n".join(lines)
    return text if len(text) <= 4000 else text[:3999] + "…"


def _approval_keyboard(token: str) -> dict[str, Any]:
    return {
        "inline_keyboard": [
            [
                {"text": "Approve", "callback_data": f"proposal:approve:{token}"},
                {"text": "Cancel", "callback_data": f"proposal:cancel:{token}"},
            ]
        ]
    }


def _followup_keyboard(note_id: str) -> dict[str, Any]:
    return {
        "inline_keyboard": [
            [
                {"text": "Done", "callback_data": f"followup:done:{note_id}"},
                {"text": "Snooze 1 day", "callback_data": f"followup:snooze:{note_id}"},
            ]
        ]
    }


class BotHandlers:
    def __init__(
        self,
        telegram: TelegramAPI,
        registry: FirestoreRegistry,
        classifier: DeepSeekClassifier,
        allowed_user_ids: frozenset[int],
        timezone_name: str,
        pending_expiry_hours: int = 24,
        birthday_reminder_days: tuple[int, ...] = (7, 1, 0),
    ) -> None:
        self.telegram = telegram
        self.registry = registry
        self.classifier = classifier
        self.allowed_user_ids = allowed_user_ids
        self.timezone_name = timezone_name
        self.pending_expiry_hours = pending_expiry_hours
        self.birthday_reminder_days = birthday_reminder_days

    def handle_update(self, update: dict[str, Any]) -> None:
        if "callback_query" in update:
            self._handle_callback(update["callback_query"])
        elif "message" in update and update["message"].get("text"):
            self._handle_message(update["message"], int(update.get("update_id", 0)))

    def _handle_message(self, message: dict[str, Any], update_id: int) -> None:
        user = message.get("from", {})
        user_id = int(user.get("id", 0))
        chat = message.get("chat", {})
        chat_id = int(chat.get("id", 0))
        text = str(message.get("text", "")).strip()
        command_word, _, body = text.partition(" ")
        command = command_word.split("@", 1)[0].casefold()
        body = body.strip()

        LOGGER.warning(
            "message_received user_id=%s chat_id=%s chat_type=%s command=%s text_len=%s update_id=%s",
            user_id,
            chat_id,
            chat.get("type"),
            command,
            len(text),
            update_id,
        )

        if command == "/whoami":
            self.telegram.send(
                chat_id,
                f"Your Telegram user ID is <code>{user_id}</code>.",
                parse_mode="HTML",
            )
            return
        if user_id not in self.allowed_user_ids or chat.get("type") != "private":
            self.telegram.send(
                chat_id, "This bot is private. Use /whoami to obtain your user ID."
            )
            return

        commands = {
            "/start": lambda: self.telegram.send(chat_id, HELP_TEXT, parse_mode="HTML"),
            "/help": lambda: self.telegram.send(chat_id, HELP_TEXT, parse_mode="HTML"),
            "/friend": lambda: self._create_entity(chat_id, user_id, "friend", body),
            "/project": lambda: self._create_entity(chat_id, user_id, "project", body),
            "/add": lambda: self._add(chat_id, user_id, body, update_id),
            "/friends": lambda: self._list_entities(chat_id, user_id, "friend"),
            "/projects": lambda: self._list_entities(chat_id, user_id, "project"),
            "/show": lambda: self._show(chat_id, user_id, body),
            "/inbox": lambda: self._inbox(chat_id, user_id),
            "/reclassify": lambda: self._reclassify(chat_id, user_id, body, update_id),
            "/followups": lambda: self._followups(chat_id, user_id),
            "/next": lambda: self._next(chat_id, user_id),
            "/done": lambda: self._done(chat_id, user_id, body),
            "/birthdays": lambda: self._birthdays(chat_id, user_id),
            "/search": lambda: self._search(chat_id, user_id, body),
        }
        handler = commands.get(command)
        if handler:
            LOGGER.warning("routing_command command=%s user_id=%s chat_id=%s", command, user_id, chat_id)
            handler()
            return
        if text and not text.startswith("/") and self._revise_pending_proposal(
            chat_id, user_id, text
        ):
            return
        else:
            self.telegram.send(chat_id, "Unknown command. Use /help.")

    def _create_entity(
        self, chat_id: int, user_id: int, target_type: str, name: str
    ) -> None:
        if not name:
            self.telegram.send(chat_id, f"Usage: /{target_type} name")
            return
        try:
            target_id, created = self.registry.upsert_entity(user_id, target_type, name)
        except ValueError as exc:
            self.telegram.send(chat_id, str(exc))
            return
        verb = "Created" if created else "Already exists"
        self.telegram.send(chat_id, f"{verb}: {target_type} {name} · {target_id[:8]}")

    def _classify(self, user_id: int, note: str) -> NoteProposal:
        return self.classifier.classify(
            note,
            self.registry.list_entity_names(user_id, "friend"),
            self.registry.list_entity_names(user_id, "project"),
            _now_in_timezone(self.timezone_name),
        )

    def _add(self, chat_id: int, user_id: int, note: str, update_id: int) -> None:
        if not note:
            self.telegram.send(chat_id, "Usage: /add your note here")
            return
        if len(note) > 5000:
            self.telegram.send(
                chat_id, "Please keep one /add note under 5,000 characters."
            )
            return
        fallback = False
        try:
            proposal = self._classify(user_id, note)
        except Exception:
            LOGGER.exception("DeepSeek classification failed")
            fallback = True
            proposal = NoteProposal(
                summary="DeepSeek was unavailable; save this to the inbox?",
                items=[
                    ProposalItem(
                        target_type="uncategorized",
                        target_name="",
                        category="general",
                        content=note,
                        occurred_on=None,
                        follow_up_at=None,
                        birthday_mm_dd=None,
                        confidence=0,
                        reason="The model request failed, so no category was guessed.",
                    )
                ],
            )
        token = safe_id(user_id, update_id, "proposal")
        token, status = self.registry.stage_pending(
            user_id,
            note,
            proposal,
            self.pending_expiry_hours,
            token=token,
        )
        if status not in {"pending"}:
            self.telegram.send(
                chat_id, f"This update was already processed ({status})."
            )
            return
        prefix = "DeepSeek categorization failed. " if fallback else ""
        sent = self.telegram.send(
            chat_id,
            prefix + _proposal_preview(proposal, self.timezone_name),
            parse_mode="HTML",
            reply_markup=_approval_keyboard(token),
        )
        result = sent.get("result") if isinstance(sent, dict) else None
        if isinstance(result, dict):
            message_id = result.get("message_id")
            if isinstance(message_id, int):
                self.registry.set_pending_message(user_id, token, chat_id, message_id)

    def _revise_pending_proposal(
        self, chat_id: int, user_id: int, instruction: str
    ) -> bool:
        LOGGER.warning(
            "revise_pending_start user_id=%s chat_id=%s instruction_len=%s",
            user_id,
            chat_id,
            len(instruction),
        )
        pending = self.registry.latest_pending_action(user_id)
        if not pending:
            LOGGER.warning("revise_pending_no_pending user_id=%s chat_id=%s", user_id, chat_id)
            return False
        if not instruction.strip():
            LOGGER.warning("revise_pending_empty_instruction user_id=%s chat_id=%s", user_id, chat_id)
            return False
        expires_at = pending.get("expires_at")
        if isinstance(expires_at, datetime) and expires_at <= utc_now():
            LOGGER.warning(
                "revise_pending_expired user_id=%s chat_id=%s token=%s",
                user_id,
                chat_id,
                pending.get("id"),
            )
            self.telegram.send(
                chat_id,
                "The latest proposal expired. Send /add again to start over.",
            )
            return True
        try:
            current_proposal = NoteProposal.model_validate(pending["proposal"])
        except Exception:
            LOGGER.exception("Stored pending proposal is invalid")
            self.telegram.send(
                chat_id, "That proposal can no longer be edited. Please cancel it."
            )
            return True
        LOGGER.warning(
            "revise_pending_loaded token=%s item_count=%s message_id=%s",
            pending.get("id"),
            len(current_proposal.items),
            pending.get("message_id"),
        )
        try:
            revised = self.classifier.revise(
                instruction,
                current_proposal,
                self.registry.list_entity_names(user_id, "friend"),
                self.registry.list_entity_names(user_id, "project"),
                _now_in_timezone(self.timezone_name),
            )
        except Exception:
            LOGGER.exception("DeepSeek revision failed")
            self.telegram.send(
                chat_id,
                "I could not revise that proposal. Please try again or press Cancel.",
            )
            return True
        LOGGER.warning(
            "revise_pending_classified token=%s revised_items=%s summary=%s",
            pending.get("id"),
            len(revised.items),
            revised.summary,
        )

        token = str(pending.get("id", ""))
        if not token:
            LOGGER.warning("revise_pending_missing_token user_id=%s chat_id=%s", user_id, chat_id)
            return False
        status = self.registry.revise_pending(user_id, token, revised, instruction)
        LOGGER.warning(
            "revise_pending_firestore_result token=%s status=%s",
            token,
            status,
        )
        if status != "pending":
            self.telegram.send(
                chat_id, f"This proposal is {status}; it was not changed."
            )
            return True

        preview = _proposal_preview(revised, self.timezone_name)
        message_id = pending.get("message_id")
        if isinstance(message_id, int):
            LOGGER.warning(
                "revise_pending_editing_existing_message token=%s message_id=%s",
                token,
                message_id,
            )
            self.telegram.edit(
                chat_id,
                message_id,
                preview,
                parse_mode="HTML",
                reply_markup=_approval_keyboard(token),
            )
        else:
            LOGGER.warning(
                "revise_pending_sending_new_message token=%s has_message_id=%s",
                token,
                message_id,
            )
            sent = self.telegram.send(
                chat_id,
                preview,
                parse_mode="HTML",
                reply_markup=_approval_keyboard(token),
            )
            result = sent.get("result") if isinstance(sent, dict) else None
            if isinstance(result, dict):
                message_id = result.get("message_id")
                if isinstance(message_id, int):
                    LOGGER.warning(
                        "revise_pending_saved_new_message_id token=%s message_id=%s",
                        token,
                        message_id,
                    )
                    self.registry.set_pending_message(user_id, token, chat_id, message_id)
        return True

    def _list_entities(self, chat_id: int, user_id: int, target_type: str) -> None:
        rows = self.registry.list_entities(user_id, target_type)
        if not rows:
            self.telegram.send(chat_id, f"No {target_type}s saved yet.")
            return
        lines = [f"<b>Your {target_type}s</b>"]
        for row in rows[:100]:
            birthday = (
                f" · birthday {row['birthday_mm_dd']}"
                if row.get("birthday_mm_dd")
                else ""
            )
            lines.append(
                f"• {html.escape(str(row['name']))} · {int(row.get('note_count', 0))} note(s){birthday}"
            )
        self.telegram.send(chat_id, "\n".join(lines), parse_mode="HTML")

    def _show(self, chat_id: int, user_id: int, body: str) -> None:
        parts = body.split(maxsplit=1)
        if len(parts) != 2 or parts[0].casefold() not in {"friend", "project"}:
            self.telegram.send(
                chat_id, "Usage: /show friend Alice (or /show project Portfolio)"
            )
            return
        target_type, name = parts[0].casefold(), parts[1]
        entity, notes = self.registry.get_entity_notes(user_id, target_type, name)
        if entity is None:
            self.telegram.send(chat_id, f"No {target_type} named {name!r}.")
            return
        lines = [f"<b>{html.escape(str(entity['name']))}</b> · {target_type}"]
        if entity.get("birthday_mm_dd"):
            lines.append(f"Birthday: {entity['birthday_mm_dd']}")
        for note in notes:
            status = (
                f" · {note['follow_up_status']}" if note.get("follow_up_status") else ""
            )
            lines.append(
                f"\n<b>#{_display_id(note['id'])}</b> · "
                f"<code>{html.escape(str(note.get('category', 'general')))}</code>{status}\n"
                f"{html.escape(_short(str(note.get('content', '')), 500))}"
            )
        if not notes:
            lines.append("No notes yet.")
        self.telegram.send(chat_id, "\n".join(lines), parse_mode="HTML")

    def _inbox(self, chat_id: int, user_id: int) -> None:
        rows = self.registry.list_uncategorized(user_id)
        if not rows:
            self.telegram.send(chat_id, "Your uncategorized inbox is empty.")
            return
        lines = ["<b>Uncategorized inbox</b>"]
        for row in rows:
            lines.append(
                f"\n<b>#{_display_id(row['id'])}</b> · "
                f"{html.escape(_short(str(row.get('content', ''))))}"
            )
        lines.append("\nRetry one with <code>/reclassify ID optional context</code>.")
        self.telegram.send(chat_id, "\n".join(lines), parse_mode="HTML")

    def _reclassify(
        self, chat_id: int, user_id: int, body: str, update_id: int
    ) -> None:
        parts = body.split(maxsplit=1)
        if not parts:
            self.telegram.send(chat_id, "Usage: /reclassify ID optional context")
            return
        note_id = self.registry.resolve_note_id(user_id, parts[0])
        if not note_id:
            self.telegram.send(chat_id, "That note ID was not found or is ambiguous.")
            return
        source = self.registry.get_uncategorized(user_id, note_id)
        if source is None:
            self.telegram.send(chat_id, "That uncategorized note was not found.")
            return
        extra = parts[1] if len(parts) == 2 else ""
        text = str(source["content"]) + (
            f"\nClarifying context: {extra}" if extra else ""
        )
        try:
            proposal = self._classify(user_id, text)
        except Exception:
            LOGGER.exception("Inbox reclassification failed")
            self.telegram.send(chat_id, "DeepSeek failed; the inbox note is unchanged.")
            return
        token = safe_id(user_id, update_id, "reclassify")
        token, status = self.registry.stage_pending(
            user_id,
            str(source.get("raw_input", source["content"])),
            proposal,
            self.pending_expiry_hours,
            token=token,
            source_note_id=note_id,
        )
        if status != "pending":
            self.telegram.send(
                chat_id, f"This update was already processed ({status})."
            )
            return
        sent = self.telegram.send(
            chat_id,
            _proposal_preview(proposal, self.timezone_name),
            parse_mode="HTML",
            reply_markup=_approval_keyboard(token),
        )
        result = sent.get("result") if isinstance(sent, dict) else None
        if isinstance(result, dict):
            message_id = result.get("message_id")
            if isinstance(message_id, int):
                self.registry.set_pending_message(user_id, token, chat_id, message_id)

    def _followups(self, chat_id: int, user_id: int) -> None:
        rows = self.registry.pending_followups(user_id)
        if not rows:
            self.telegram.send(chat_id, "No pending follow-ups or next actions.")
            return
        lines = ["<b>Pending follow-ups and next actions</b>"]
        for row in rows:
            lines.append(
                f"\n<b>#{_display_id(row['id'])}</b> · "
                f"{html.escape(str(row.get('target_name', 'Inbox')))} · "
                f"{_local_datetime(row.get('follow_up_at'), self.timezone_name)}\n"
                f"{html.escape(_short(str(row.get('content', ''))))}"
            )
        lines.append("\nComplete one with <code>/done ID</code>.")
        self.telegram.send(chat_id, "\n".join(lines), parse_mode="HTML")

    def _next(self, chat_id: int, user_id: int) -> None:
        rows = self.registry.project_next_actions(user_id)
        if not rows:
            self.telegram.send(chat_id, "No open project next actions.")
            return
        lines = ["<b>Project next actions</b>"]
        for row in rows:
            lines.append(
                f"\n<b>#{_display_id(row['id'])}</b> · "
                f"{html.escape(str(row.get('target_name', 'Project')))}\n"
                f"{html.escape(_short(str(row.get('content', ''))))}"
            )
        self.telegram.send(chat_id, "\n".join(lines), parse_mode="HTML")

    def _done(self, chat_id: int, user_id: int, value: str) -> None:
        note_id = self.registry.resolve_note_id(user_id, value) if value else None
        if not note_id:
            self.telegram.send(
                chat_id, "Usage: /done ID (the ID was not found or was ambiguous)"
            )
            return
        changed = self.registry.complete_followup(user_id, note_id)
        self.telegram.send(
            chat_id,
            f"Completed #{_display_id(note_id)}."
            if changed
            else "That item is no longer pending.",
        )

    def _birthdays(self, chat_id: int, user_id: int) -> None:
        rows = [
            row
            for row in self.registry.list_entities(user_id, "friend")
            if row.get("birthday_mm_dd")
        ]
        if not rows:
            self.telegram.send(chat_id, "No birthdays saved yet. Add one with /add.")
            return
        rows.sort(key=lambda row: (row["birthday_mm_dd"], str(row["name"]).casefold()))
        lines = ["<b>Saved birthdays</b>"] + [
            f"• {row['birthday_mm_dd']} · {html.escape(str(row['name']))}"
            for row in rows
        ]
        self.telegram.send(chat_id, "\n".join(lines), parse_mode="HTML")

    def _search(self, chat_id: int, user_id: int, query: str) -> None:
        if not query:
            self.telegram.send(chat_id, "Usage: /search words")
            return
        rows = self.registry.search_notes(user_id, query)
        if not rows:
            self.telegram.send(chat_id, "No matching notes.")
            return
        lines = [f"<b>Results for {html.escape(query)}</b>"]
        for row in rows:
            lines.append(
                f"\n<b>#{_display_id(row['id'])}</b> · "
                f"{html.escape(str(row.get('target_name', 'Inbox')))} · "
                f"<code>{html.escape(str(row.get('category', 'general')))}</code>\n"
                f"{html.escape(_short(str(row.get('content', ''))))}"
            )
        self.telegram.send(chat_id, "\n".join(lines), parse_mode="HTML")

    def _handle_callback(self, query: dict[str, Any]) -> None:
        user_id = int(query.get("from", {}).get("id", 0))
        callback_id = str(query.get("id", ""))
        message = query.get("message", {})
        chat_id = int(message.get("chat", {}).get("id", 0))
        message_id = int(message.get("message_id", 0))
        if user_id not in self.allowed_user_ids:
            self.telegram.answer_callback(callback_id, "This bot is private.")
            return
        self.telegram.answer_callback(callback_id)
        parts = str(query.get("data", "")).split(":", 2)
        if len(parts) != 3:
            self.telegram.edit(chat_id, message_id, "Invalid action.")
            return
        kind, action, value = parts
        if kind == "proposal":
            if action == "approve":
                status, note_ids = self.registry.approve_pending(user_id, value)
                if status == "approved":
                    ids = ", ".join(f"#{_display_id(note_id)}" for note_id in note_ids)
                    self.telegram.edit(
                        chat_id, message_id, f"Saved {len(note_ids)} note(s): {ids}"
                    )
                    return
            else:
                status = self.registry.cancel_pending(user_id, value)
                if status == "cancelled":
                    self.telegram.edit(
                        chat_id, message_id, "Cancelled. Nothing was saved."
                    )
                    return
            self.telegram.edit(
                chat_id, message_id, f"This proposal is {status}; it was not changed."
            )
            return
        if kind == "followup":
            if action == "done":
                changed = self.registry.complete_followup(user_id, value)
                text = (
                    f"Completed #{_display_id(value)}."
                    if changed
                    else "This item is no longer pending."
                )
            else:
                changed = self.registry.snooze_followup(
                    user_id, value, utc_now() + timedelta(days=1)
                )
                text = (
                    f"Snoozed #{_display_id(value)} for one day."
                    if changed
                    else "This item is no longer pending."
                )
            self.telegram.edit(chat_id, message_id, text)

    def send_daily_reminders(self) -> None:
        local_now = datetime.now(ZoneInfo(self.timezone_name))
        today = local_now.date()
        for user_id in self.allowed_user_ids:
            for friend in self.registry.birthdays_at_offsets(
                user_id, today, self.birthday_reminder_days
            ):
                reference = f"{friend['id']}:{friend['days_until']}"
                if self.registry.has_notification(
                    user_id, "birthday", reference, today
                ):
                    continue
                days = friend["days_until"]
                text = (
                    f"🎂 Today is {friend['name']}'s birthday."
                    if days == 0
                    else f"🎂 {friend['name']}'s birthday is in {days} day(s), on {friend['next_birthday']}."
                )
                self.telegram.send(user_id, text)
                self.registry.mark_notification(user_id, "birthday", reference, today)

            for note in self.registry.due_followups(user_id):
                reference = str(note["id"])
                if self.registry.has_notification(
                    user_id, "followup", reference, today
                ):
                    continue
                self.telegram.send(
                    user_id,
                    f"⏰ Follow-up #{_display_id(note['id'])} · "
                    f"{note.get('target_name', 'Inbox')}\n{_short(str(note.get('content', '')), 800)}",
                    reply_markup=_followup_keyboard(note["id"]),
                )
                self.registry.mark_notification(user_id, "followup", reference, today)

            inbox = self.registry.list_uncategorized(user_id, limit=1000)
            if inbox and not self.registry.has_notification(
                user_id, "inbox", "daily", today
            ):
                self.telegram.send(
                    user_id,
                    f"📥 You have {len(inbox)} uncategorized note(s). Use /inbox to review them.",
                )
                self.registry.mark_notification(user_id, "inbox", "daily", today)
