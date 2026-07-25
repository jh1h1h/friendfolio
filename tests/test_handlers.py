from __future__ import annotations

import sys
import unittest
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).parents[1] / "functions"))

from friendfolio.handlers import BotHandlers, _is_context_note  # noqa: E402
from friendfolio.models import (  # noqa: E402
    ContextSelection,
    NoteProposal,
    SearchAnswer,
    SearchPlan,
)

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
        self.search_calls: list[dict[str, Any]] = []
        self.search_results: list[dict[str, Any]] = [
            {
                "id": "note1234",
                "target_name": "Alice",
                "category": "note",
                "content": "Alice started a new role.",
            }
        ]
        self.context_requests: list[tuple[int, str, str]] = []
        self.confidence_threshold = 0.65
        self.approval_results: list[dict[str, str]] = [
            {
                "id": "note1234",
                "action": "updated",
                "target_name": "Alice",
                "target_type": "friend",
                "record_type": "note",
                "content": "Alice likes hiking and started a new role.",
            }
        ]

    def list_entity_names(self, user_id: int, target_type: str) -> list[str]:
        del user_id
        return ["Alice"] if target_type == "friend" else []

    def get_entity_notes(
        self, user_id: int, target_type: str, name: str
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        self.context_requests.append((user_id, target_type, name))
        return (
            {"name": name},
            [
                {
                    "category": "note",
                    "record_type": "note",
                    "content": "Alice likes hiking.",
                    "follow_up_at": datetime(2026, 7, 24, 13, 0, tzinfo=UTC),
                },
                {
                    "category": "note",
                    "record_type": "history",
                    "content": "Alice went hiking today.",
                },
            ],
        )

    def get_confidence_threshold(self, user_id: int) -> float:
        del user_id
        return self.confidence_threshold

    def set_confidence_threshold(self, user_id: int, value: float) -> None:
        del user_id
        self.confidence_threshold = value

    def resolve_note_id(self, user_id: int, value: str) -> str | None:
        del user_id
        return "inbox-note-1234" if value == "inbox123" else None

    def get_uncategorized(
        self, user_id: int, note_id: str
    ) -> dict[str, Any] | None:
        del user_id
        if note_id != "inbox-note-1234":
            return None
        return {
            "id": note_id,
            "content": "Alice started a new role.",
            "raw_input": "Alice started a new role.",
        }

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

    def approve_pending(
        self, owner_user_id: int, token: str
    ) -> tuple[str, list[dict[str, str]]]:
        del owner_user_id, token
        return "approved", self.approval_results

    def search_notes(
        self,
        owner_user_id: int,
        query: str,
        limit: int = 20,
        plan: SearchPlan | None = None,
    ) -> list[dict[str, Any]]:
        self.search_calls.append(
            {"owner_user_id": owner_user_id, "query": query, "limit": limit, "plan": plan}
        )
        return self.search_results[:limit]


class FakeClassifier:
    def __init__(self) -> None:
        self.prior_context: list[dict[str, Any]] = []
        self.confidence_threshold = 0.0
        self.search_context: list[dict[str, Any]] = []
        self.classification_error = False

    def select_context(self, *args: Any, **kwargs: Any) -> ContextSelection:
        del args
        trace = kwargs.get("trace")
        if trace:
            trace(
                {
                    "stage": "context_selection",
                    "event": "response",
                    "raw_response": '{"friend_names":["Alice"],"project_names":[]}',
                }
            )
        return ContextSelection(friend_names=["Alice"], project_names=[])

    def classify(self, *args: Any, **kwargs: Any) -> NoteProposal:
        trace = kwargs.get("trace")
        if trace:
            trace(
                {
                    "stage": "proposal",
                    "event": "response",
                    "raw_response": '{"summary":"Save Alice"}',
                }
            )
        if self.classification_error:
            raise RuntimeError("simulated DeepSeek failure")
        self.prior_context = args[4]
        self.confidence_threshold = args[5]
        return NoteProposal.model_validate(proposal_payload())

    def revise(self, instruction: str, current_proposal: NoteProposal, *args: Any, **kwargs: Any) -> NoteProposal:
        del args
        trace = kwargs.get("trace")
        if trace:
            trace(
                {
                    "stage": "proposal_revision",
                    "event": "response",
                    "raw_response": '{"summary":"Revised proposal"}',
                }
            )
        payload = current_proposal.model_dump(mode="json")
        payload["summary"] = f"Revised: {instruction}"
        return NoteProposal.model_validate(payload)

    def search(self, query: str, *args: Any, **kwargs: Any) -> SearchPlan:
        del args, kwargs
        return SearchPlan.model_validate(
            {
                "summary": f"Search for {query}",
                "include_terms": ["alice", "role"],
                "exclude_terms": [],
                "entity_names": ["Alice"],
                "target_types": ["friend"],
                "categories": ["note"],
                "limit": 20,
                "sort_by": "relevance",
                "require_all_terms": False,
            }
        )

    def answer_search(
        self, query: str, matching_notes: list[dict[str, Any]], *args: Any
    ) -> SearchAnswer:
        del query, args
        self.search_context = matching_notes
        return SearchAnswer(answer="Alice started a new role.")


class HandlerTests(unittest.TestCase):
    def test_context_note_filter_distinguishes_friend_and_project_records(self):
        self.assertTrue(_is_context_note({"record_type": "note"}, "friend"))
        self.assertTrue(_is_context_note({"record_type": "summary"}, "friend"))
        self.assertFalse(_is_context_note({"record_type": "history"}, "friend"))
        self.assertTrue(_is_context_note({"record_type": "note"}, "project"))

    def build(self) -> tuple[BotHandlers, FakeTelegram, FakeRegistry]:
        telegram = FakeTelegram()
        registry = FakeRegistry()
        classifier = FakeClassifier()
        handlers = BotHandlers(
            telegram=telegram,  # type: ignore[arg-type]
            registry=registry,  # type: ignore[arg-type]
            classifier=classifier,  # type: ignore[arg-type]
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
        self.assertEqual(registry.context_requests, [(123, "friend", "Alice")])
        self.assertEqual(
            handlers.classifier.prior_context[0]["notes"][0]["content"],  # type: ignore[attr-defined]
            "Alice likes hiking.",
        )
        self.assertEqual(len(handlers.classifier.prior_context[0]["notes"]), 1)  # type: ignore[attr-defined]
        self.assertEqual(
            handlers.classifier.prior_context[0]["notes"][0]["follow_up_at"],  # type: ignore[attr-defined]
            "2026-07-24T13:00:00+00:00",
        )
        self.assertEqual(handlers.classifier.confidence_threshold, 0.65)  # type: ignore[attr-defined]
        self.assertIn("Proposed update", telegram.sent[0]["text"])
        buttons = telegram.sent[0]["reply_markup"]["inline_keyboard"][0]
        self.assertTrue(buttons[0]["callback_data"].startswith("proposal:approve:"))
        self.assertLessEqual(len(buttons[0]["callback_data"]), 64)

    def test_add_debug_returns_trace_and_stages_proposal(self) -> None:
        handlers, telegram, registry = self.build()
        handlers.handle_update(
            {
                "update_id": 103,
                "message": {
                    "text": "/add -debug Alice started a new role",
                    "from": {"id": 123},
                    "chat": {"id": 123, "type": "private"},
                },
            }
        )

        self.assertEqual(len(registry.staged), 1)
        self.assertTrue(registry.staged[0]["debug_mode"])
        self.assertTrue(any("raw_response" in message["text"] for message in telegram.sent))
        self.assertTrue(any("Proposed update" in message["text"] for message in telegram.sent))

    def test_add_debug_includes_errors_without_staging(self) -> None:
        handlers, telegram, registry = self.build()
        handlers.classifier.classification_error = True  # type: ignore[attr-defined]
        handlers.handle_update(
            {
                "update_id": 104,
                "message": {
                    "text": "/add -debug Alice update",
                    "from": {"id": 123},
                    "chat": {"id": 123, "type": "private"},
                },
            }
        )

        self.assertEqual(registry.staged, [])
        self.assertTrue(any("simulated DeepSeek failure" in item["text"] for item in telegram.sent))

    def test_reclassify_debug_returns_trace_and_stages_proposal(self) -> None:
        handlers, telegram, registry = self.build()
        handlers.handle_update(
            {
                "update_id": 105,
                "message": {
                    "text": "/reclassify -debug inbox123 extra context",
                    "from": {"id": 123},
                    "chat": {"id": 123, "type": "private"},
                },
            }
        )

        self.assertEqual(len(registry.staged), 1)
        self.assertTrue(registry.staged[0]["debug_mode"])
        self.assertTrue(any("context_selection" in item["text"] for item in telegram.sent))
        self.assertTrue(any('"stage": "proposal"' in item["text"] for item in telegram.sent))

    def test_confidence_command_views_and_updates_threshold(self) -> None:
        handlers, telegram, registry = self.build()
        for update_id, text in enumerate(("/confidence", "/confidence 75"), 1):
            handlers.handle_update(
                {
                    "update_id": update_id,
                    "message": {
                        "text": text,
                        "from": {"id": 123},
                        "chat": {"id": 123, "type": "private"},
                    },
                }
            )

        self.assertIn("65%", telegram.sent[0]["text"])
        self.assertEqual(registry.confidence_threshold, 0.75)
        self.assertIn("75%", telegram.sent[1]["text"])

    def test_approval_confirmation_shows_action_target_and_content(self) -> None:
        handlers, telegram, _ = self.build()
        handlers.handle_update(
            {
                "callback_query": {
                    "id": "callback-1",
                    "from": {"id": 123},
                    "data": "proposal:approve:token-1",
                    "message": {
                        "message_id": 42,
                        "chat": {"id": 123, "type": "private"},
                    },
                }
            }
        )

        self.assertIn("Updated Alice’s note", telegram.edits[0]["text"])
        self.assertIn("likes hiking and started a new role", telegram.edits[0]["text"])
        self.assertEqual(telegram.edits[0]["parse_mode"], "HTML")

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

    def test_debug_proposal_automatically_traces_and_applies_steering(self) -> None:
        handlers, telegram, registry = self.build()
        handlers.handle_update(
            {
                "update_id": 106,
                "message": {
                    "text": "/add -debug Alice started a new role",
                    "from": {"id": 123},
                    "chat": {"id": 123, "type": "private"},
                },
            }
        )
        handlers.handle_update(
            {
                "update_id": 107,
                "message": {
                    "text": "make it about her manager role",
                    "from": {"id": 123},
                    "chat": {"id": 123, "type": "private"},
                },
            }
        )

        self.assertEqual(
            registry.staged[0]["last_instruction"],
            "make it about her manager role",
        )
        self.assertIn("Revised:", telegram.edits[0]["text"])
        self.assertTrue(
            any("proposal_revision" in message["text"] for message in telegram.sent)
        )

    def test_search_uses_deepseek_plan(self) -> None:
        handlers, telegram, registry = self.build()
        handlers.handle_update(
            {
                "update_id": 102,
                "message": {
                    "text": "/search find Alice's role",
                    "from": {"id": 123},
                    "chat": {"id": 123, "type": "private"},
                },
            }
        )
        self.assertEqual(len(registry.search_calls), 1)
        self.assertIsNotNone(registry.search_calls[0]["plan"])
        self.assertEqual(
            handlers.classifier.search_context[0]["content"],  # type: ignore[attr-defined]
            "Alice started a new role.",
        )
        self.assertIn("<b>Answer</b>", telegram.sent[-1]["text"])
        self.assertIn("Alice", telegram.sent[-1]["text"])

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
