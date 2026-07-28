from __future__ import annotations

import json
import sys
import unittest
from datetime import date, datetime
from pathlib import Path

import httpx
from pydantic import ValidationError


sys.path.insert(0, str(Path(__file__).parents[1] / "functions"))

from friendfolio.deepseek import API_URL, DeepSeekClassifier  # noqa: E402
from friendfolio.models import (  # noqa: E402
    ContextSelection,
    NoteProposal,
    ProposalItem,
    SearchAnswer,
)
from friendfolio.store import FirestoreRegistry, entity_id, normalize_name, safe_id  # noqa: E402
from scripts.set_webhook import COMMANDS  # noqa: E402


def proposal_payload() -> dict[str, object]:
    return {
        "summary": "Save Alice's new role",
        "items": [
            {
                "target_type": "friend",
                "target_name": "Alice",
                "category": "note",
                "content": "Alice started a new role.",
                "occurred_on": None,
                "follow_up_at": None,
                "birthday_mm_dd": None,
                "confidence": 0.97,
                "reason": "Alice is explicitly named.",
            }
        ],
    }


class ModelAndHelperTests(unittest.TestCase):
    def test_telegram_menu_registers_every_implemented_command(self) -> None:
        registered = {item["command"] for item in COMMANDS}
        self.assertEqual(
            registered,
            {
                "start",
                "help",
                "whoami",
                "add",
                "friend",
                "project",
                "friends",
                "projects",
                "show",
                "inbox",
                "reclassify",
                "followups",
                "next",
                "done",
                "birthdays",
                "search",
                "confidence",
            },
        )

    def test_entity_ids_are_case_and_whitespace_insensitive(self) -> None:
        self.assertEqual(normalize_name("  Alice   TAN "), "alice tan")
        self.assertEqual(entity_id("Alice Tan"), entity_id(" alice   TAN "))
        self.assertEqual(len(safe_id(1, "proposal", 2)), 32)

    def test_invalid_project_birthday_is_rejected(self) -> None:
        payload = proposal_payload()["items"][0].copy()  # type: ignore[index,union-attr]
        payload.update(
            target_type="project",
            target_name="Portfolio",
            category="note",
            birthday_mm_dd="02-30",
        )
        with self.assertRaises(ValidationError):
            ProposalItem.model_validate(payload)

    def test_uncategorized_item_is_normalized(self) -> None:
        payload = proposal_payload()["items"][0].copy()  # type: ignore[index,union-attr]
        payload.update(
            target_type="uncategorized", target_name="Guess", category="note"
        )
        item = ProposalItem.model_validate(payload)
        self.assertEqual(item.target_name, "")
        self.assertEqual(item.category, "note")

    def test_follow_up_requires_a_scheduled_time(self) -> None:
        payload = proposal_payload()["items"][0].copy()  # type: ignore[index,union-attr]
        payload["category"] = "follow_up"
        with self.assertRaises(ValidationError):
            ProposalItem.model_validate(payload)

    def test_friend_follow_up_proposal_also_requires_current_note(self) -> None:
        payload = proposal_payload()
        payload["items"][0].update(  # type: ignore[index]
            category="follow_up",
            follow_up_at="2026-07-26T09:00:00+08:00",
        )
        with self.assertRaises(ValidationError):
            NoteProposal.model_validate(payload)

    def test_project_follow_up_proposal_also_requires_current_note(self) -> None:
        payload = proposal_payload()
        payload["items"][0].update(  # type: ignore[index]
            target_type="project",
            target_name="Portfolio",
            category="follow_up",
            follow_up_at="2026-07-26T09:00:00+08:00",
        )
        with self.assertRaises(ValidationError):
            NoteProposal.model_validate(payload)

    def test_missing_friend_sections_are_reported_without_changing_content(self) -> None:
        payload = proposal_payload()
        proposal = NoteProposal.model_validate(payload)

        self.assertEqual(
            DeepSeekClassifier.missing_note_sections(proposal)["Alice"],
            list(DeepSeekClassifier.NOTE_SECTIONS),
        )
        self.assertEqual(proposal.items[0].content, "Alice started a new role.")

    def test_deepseek_request_uses_v4_flash_json_mode(self) -> None:
        captured: dict[str, object] = {}

        def responder(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": json.dumps(proposal_payload())}}
                    ]
                },
            )

        classifier = DeepSeekClassifier(
            "test-key", "deepseek-v4-flash", "Asia/Singapore"
        )
        classifier.client = httpx.Client(transport=httpx.MockTransport(responder))
        result = classifier.classify(
            "Alice started a new role",
            [],
            [],
            datetime.now(),
            [{"follow_up_at": "2026-07-24T13:00:00+00:00"}],
        )

        self.assertEqual(result.items[0].target_name, "Alice")
        self.assertEqual(result.items[0].content, "Alice started a new role.")
        payload = json.loads(captured["messages"][1]["content"])  # type: ignore[index]
        self.assertEqual(
            payload["prior_context"][0]["follow_up_at"], "2026-07-24T13:00:00+00:00"
        )
        system_prompt = captured["messages"][0]["content"]  # type: ignore[index]
        self.assertIn("friend or project affected", system_prompt)
        self.assertIn("append-only history", system_prompt)
        self.assertIn("Format only friend", system_prompt)
        self.assertIn("clearly named in new_note is a friend target", system_prompt)
        self.assertIn(
            "create a new friend note rather than update an existing friend note",
            system_prompt,
        )
        self.assertIn("relative time words", system_prompt)
        self.assertIn("age N as of YYYY-MM-DD", system_prompt)
        self.assertEqual(captured["model"], "deepseek-v4-flash")
        self.assertEqual(captured["response_format"], {"type": "json_object"})
        self.assertEqual(captured["thinking"], {"type": "disabled"})

    def test_deepseek_selects_context_before_classification(self) -> None:
        captured: dict[str, object] = {}

        def responder(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": ContextSelection(
                                    friend_names=["Alice"], project_names=[]
                                ).model_dump_json()
                            }
                        }
                    ]
                },
            )

        classifier = DeepSeekClassifier(
            "test-key", "deepseek-v4-flash", "Asia/Singapore"
        )
        classifier.client = httpx.Client(transport=httpx.MockTransport(responder))
        selection = classifier.select_context(
            "She was promoted", ["Alice"], [], __import__("datetime").datetime.now()
        )

        self.assertEqual(selection.friend_names, ["Alice"])
        payload = json.loads(captured["messages"][1]["content"])  # type: ignore[index]
        self.assertEqual(payload["existing_friends"], ["Alice"])

    def test_low_confidence_proposal_is_sent_to_inbox(self) -> None:
        payload = proposal_payload()
        payload["items"][0]["confidence"] = 0.6  # type: ignore[index]

        def responder(request: httpx.Request) -> httpx.Response:
            del request
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": json.dumps(payload)}}
                    ]
                },
            )

        classifier = DeepSeekClassifier(
            "test-key", "deepseek-v4-flash", "Asia/Singapore"
        )
        classifier.client = httpx.Client(transport=httpx.MockTransport(responder))
        result = classifier.classify(
            "Alice started a new role", [], [], datetime.now(), confidence_threshold=0.7
        )

        self.assertEqual(result.items[0].target_type, "uncategorized")
        self.assertIn("70% threshold", result.items[0].reason)

    def test_deepseek_search_request_uses_json_mode(self) -> None:
        captured: dict[str, object] = {}

        def responder(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "summary": "Search for Alice's role",
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
                            }
                        }
                    ]
                },
            )

        classifier = DeepSeekClassifier(
            "test-key", "deepseek-v4-flash", "Asia/Singapore"
        )
        classifier.client = httpx.Client(transport=httpx.MockTransport(responder))
        plan = classifier.search(
            "find Alice's role", ["Alice"], [], __import__("datetime").datetime.now()
        )

        self.assertEqual(plan.include_terms, ["alice", "role"])
        self.assertEqual(captured["model"], "deepseek-v4-flash")
        self.assertEqual(captured["response_format"], {"type": "json_object"})
        self.assertEqual(captured["thinking"], {"type": "disabled"})

    def test_deepseek_answers_search_from_matching_note_context(self) -> None:
        captured: dict[str, object] = {}

        def responder(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": SearchAnswer(
                                    answer="Alice started a new role."
                                ).model_dump_json()
                            }
                        }
                    ]
                },
            )

        classifier = DeepSeekClassifier(
            "test-key", "deepseek-v4-flash", "Asia/Singapore"
        )
        classifier.client = httpx.Client(transport=httpx.MockTransport(responder))
        answer = classifier.answer_search(
            "What changed for Alice?",
            [{"target_name": "Alice", "content": "Alice started a new role."}],
            datetime.now(),
        )

        self.assertEqual(answer.answer, "Alice started a new role.")
        payload = json.loads(captured["messages"][1]["content"])  # type: ignore[index]
        self.assertEqual(payload["matching_notes"][0]["target_name"], "Alice")

    def test_february_29_maps_to_february_28_in_non_leap_year(self) -> None:
        self.assertEqual(
            FirestoreRegistry._birthday_for_year("02-29", 2027), date(2027, 2, 28)
        )

    def test_deepseek_endpoint_is_https(self) -> None:
        self.assertEqual(API_URL, "https://api.deepseek.com/chat/completions")


if __name__ == "__main__":
    unittest.main()
