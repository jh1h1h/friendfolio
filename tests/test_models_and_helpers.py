from __future__ import annotations

import json
import sys
import unittest
from datetime import date
from pathlib import Path

import httpx
from pydantic import ValidationError


sys.path.insert(0, str(Path(__file__).parents[1] / "functions"))

from friendfolio.deepseek import API_URL, DeepSeekClassifier  # noqa: E402
from friendfolio.models import ProposalItem  # noqa: E402
from friendfolio.store import FirestoreRegistry, entity_id, normalize_name, safe_id  # noqa: E402


def proposal_payload() -> dict[str, object]:
    return {
        "summary": "Save Alice's new role",
        "items": [
            {
                "target_type": "friend",
                "target_name": "Alice",
                "category": "status",
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
    def test_entity_ids_are_case_and_whitespace_insensitive(self) -> None:
        self.assertEqual(normalize_name("  Alice   TAN "), "alice tan")
        self.assertEqual(entity_id("Alice Tan"), entity_id(" alice   TAN "))
        self.assertEqual(len(safe_id(1, "proposal", 2)), 32)

    def test_invalid_project_birthday_is_rejected(self) -> None:
        payload = proposal_payload()["items"][0].copy()  # type: ignore[index,union-attr]
        payload.update(
            target_type="project",
            target_name="Portfolio",
            category="birthday",
            birthday_mm_dd="02-30",
        )
        with self.assertRaises(ValidationError):
            ProposalItem.model_validate(payload)

    def test_uncategorized_item_is_normalized(self) -> None:
        payload = proposal_payload()["items"][0].copy()  # type: ignore[index,union-attr]
        payload.update(
            target_type="uncategorized", target_name="Guess", category="status"
        )
        item = ProposalItem.model_validate(payload)
        self.assertEqual(item.target_name, "")
        self.assertEqual(item.category, "general")

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
            "Alice started a new role", [], [], __import__("datetime").datetime.now()
        )

        self.assertEqual(result.items[0].target_name, "Alice")
        self.assertEqual(captured["model"], "deepseek-v4-flash")
        self.assertEqual(captured["response_format"], {"type": "json_object"})
        self.assertEqual(captured["thinking"], {"type": "disabled"})

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
                                        "categories": ["status"],
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

    def test_february_29_maps_to_february_28_in_non_leap_year(self) -> None:
        self.assertEqual(
            FirestoreRegistry._birthday_for_year("02-29", 2027), date(2027, 2, 28)
        )

    def test_deepseek_endpoint_is_https(self) -> None:
        self.assertEqual(API_URL, "https://api.deepseek.com/chat/completions")


if __name__ == "__main__":
    unittest.main()
