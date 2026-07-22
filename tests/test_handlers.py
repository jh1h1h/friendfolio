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

    def send(self, chat_id: int, text: str, **kwargs: Any) -> None:
        self.sent.append({"chat_id": chat_id, "text": text, **kwargs})


class FakeRegistry:
    def __init__(self) -> None:
        self.staged: list[dict[str, Any]] = []

    def list_entity_names(self, user_id: int, target_type: str) -> list[str]:
        del user_id, target_type
        return []

    def stage_pending(
        self, user_id: int, raw: str, proposal: NoteProposal, expiry: int, **kwargs: Any
    ):
        self.staged.append(
            {
                "user_id": user_id,
                "raw": raw,
                "proposal": proposal,
                "expiry": expiry,
                **kwargs,
            }
        )
        return kwargs["token"], "pending"


class FakeClassifier:
    def classify(self, *args: Any, **kwargs: Any) -> NoteProposal:
        del args, kwargs
        return NoteProposal.model_validate(proposal_payload())


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
