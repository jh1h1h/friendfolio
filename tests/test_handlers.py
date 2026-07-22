from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).parents[1] / "functions"))

from friendfolio.handlers import BotHandlers  # noqa: E402
from friendfolio.models import NoteProposal  # noqa: E402

from .test_models_and_helpers import proposal_payload  # noqa: E402


class FakeTelegram:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.edits: list[dict[str, Any]] = []
        self._message_id = 1

    def send(self, chat_id: int, text: str, **kwargs: Any) -> dict[str, Any]:
        message_id = self._message_id
        self._message_id += 1
        payload = {"chat_id": chat_id, "text": text, "message_id": message_id, **kwargs}
        self.sent.append(payload)
        return {"ok": True, "result": {"message_id": message_id}}

    def edit(self, chat_id: int, message_id: int, text: str, **kwargs: Any) -> dict[str, Any]:
        payload = {"chat_id": chat_id, "message_id": message_id, "text": text, **kwargs}
        self.edits.append(payload)
        return {"ok": True, "result": {"message_id": message_id}}

    def answer_callback(self, callback_query_id: str, text: str | None = None) -> None:
        del callback_query_id, text


class FakeRegistry:
    def __init__(self) -> None:
        self.staged: list[dict[str, Any]] = []
        self.pending: dict[str, dict[str, Any]] = {}

    def list_entity_names(self, user_id: int, target_type: str) -> list[str]:
        del user_id, target_type
        return []

    def stage_pending(
        self, user_id: int, raw: str, proposal: NoteProposal, expiry: int, **kwargs: Any
    ):
        item = {
                "user_id": user_id,
                "raw": raw,
                "proposal": proposal,
                "expiry": expiry,
                **kwargs,
                "status": "pending",
            }
        self.staged.append(item)
        self.pending[kwargs["token"]] = item
        return kwargs["token"], "pending"

    def latest_pending_action(self, user_id: int) -> dict[str, Any] | None:
        del user_id
        for item in reversed(self.staged):
            if item.get("status") == "pending":
                return {**item, "id": item["token"]}
        return None

    def set_pending_message(self, user_id: int, token: str, chat_id: int, message_id: int) -> bool:
        del user_id
        item = self.pending.get(token)
        if not item:
            return False
        item.update({"chat_id": chat_id, "message_id": message_id})
        return True

    def revise_pending(
        self, user_id: int, token: str, proposal: NoteProposal, instruction: str
    ) -> str:
        del user_id
        item = self.pending.get(token)
        if not item:
            return "missing"
        item.update({"proposal": proposal, "last_instruction": instruction})
        return "pending"

    def cancel_pending(self, owner_user_id: int, token: str) -> str:
        del owner_user_id
        item = self.pending.get(token)
        if not item:
            return "missing"
        item["status"] = "cancelled"
        return "cancelled"


class FakeClassifier:
    def classify(self, *args: Any, **kwargs: Any) -> NoteProposal:
        del args, kwargs
        return NoteProposal.model_validate(proposal_payload())

    def revise(self, instruction: str, current_proposal: NoteProposal, *args: Any, **kwargs: Any) -> NoteProposal:
        del args, kwargs
        payload = current_proposal.model_dump(mode="json")
        payload["summary"] = f"Revised: {instruction}"
        return NoteProposal.model_validate(payload)


class HandlerTests(unittest.TestCase):
    def build(self) -> tuple[BotHandlers, FakeTelegram, FakeRegistry]:
        telegram = FakeTelegram()
        registry = FakeRegistry()
        handlers = BotHandlers(
            telegram=telegram,  # type: ignore[arg-type]
            registry=registry,  # type: ignore[arg-type]
            classifier=FakeClassifier(),  # type: ignore[arg-type]
            allowed_user_ids=frozenset({123}),
            timezone_name="Asia/Singapore",
        )
        return handlers, telegram, registry

    def test_add_stages_proposal_and_sends_approval_buttons(self) -> None:
        handlers, telegram, registry = self.build()
        handlers.handle_update(
            {
                "update_id": 99,
                "message": {
                    "text": "/add Alice started a new role",
                    "from": {"id": 123},
                    "chat": {"id": 123, "type": "private"},
                },
            }
        )
        self.assertEqual(len(registry.staged), 1)
        self.assertIn("Proposed update", telegram.sent[0]["text"])
        buttons = telegram.sent[0]["reply_markup"]["inline_keyboard"][0]
        self.assertTrue(buttons[0]["callback_data"].startswith("proposal:approve:"))
        self.assertLessEqual(len(buttons[0]["callback_data"]), 64)

    def test_free_text_revises_the_pending_proposal(self) -> None:
        handlers, telegram, registry = self.build()
        handlers.handle_update(
            {
                "update_id": 99,
                "message": {
                    "text": "/add Alice started a new role",
                    "from": {"id": 123},
                    "chat": {"id": 123, "type": "private"},
                },
            }
        )
        handlers.handle_update(
            {
                "update_id": 100,
                "message": {
                    "text": "make it about her new manager role",
                    "from": {"id": 123},
                    "chat": {"id": 123, "type": "private"},
                },
            }
        )
        self.assertEqual(len(telegram.edits), 1)
        self.assertIn("Revised:", telegram.edits[0]["text"])
        self.assertEqual(registry.staged[0]["last_instruction"], "make it about her new manager role")
        self.assertIn("Approve", telegram.edits[0]["reply_markup"]["inline_keyboard"][0][0]["text"])

    def test_unauthorized_user_can_only_get_whoami(self) -> None:
        handlers, telegram, registry = self.build()
        handlers.handle_update(
            {
                "update_id": 100,
                "message": {
                    "text": "/add private note",
                    "from": {"id": 999},
                    "chat": {"id": 999, "type": "private"},
                },
            }
        )
        self.assertEqual(registry.staged, [])
        self.assertIn("private", telegram.sent[0]["text"])

        handlers.handle_update(
            {
                "update_id": 101,
                "message": {
                    "text": "/whoami",
                    "from": {"id": 999},
                    "chat": {"id": 999, "type": "private"},
                },
            }
        )
        self.assertIn("999", telegram.sent[-1]["text"])


if __name__ == "__main__":
    unittest.main()
