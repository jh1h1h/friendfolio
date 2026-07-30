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
from friendfolio.errors import NoteOperationError  # noqa: E402
from friendfolio.migrations import migrate_friend_note  # noqa: E402
from friendfolio.note_schema import (  # noqa: E402
    FRIEND_NOTE_SCHEMA_VERSION,
    FRIEND_NOTE_SECTIONS,
)
from friendfolio.models import (  # noqa: E402
    ContextSelection,
    NoteEditOperation,
    NoteProposal,
    OperationProposal,
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


def operation_payload(
    action: str = "append",
    *,
    section: str = "Current events",
    match: str | None = None,
    content: str | None = "Alice started a new role.",
    source_quote: str = "Alice started a new role",
) -> dict[str, object]:
    return {
        "summary": "Update Alice's note",
        "items": [
            {
                "target_type": "friend",
                "target_name": "Alice",
                "category": "note",
                "operations": [
                    {
                        "action": action,
                        "section": section,
                        "match": match,
                        "content": content,
                        "source_quote": source_quote,
                        "reason": "The source explicitly supports this edit.",
                    }
                ],
                "content": None,
                "occurred_on": None,
                "follow_up_at": None,
                "birthday_mm_dd": None,
                "confidence": 0.97,
                "reason": "Alice is explicitly named.",
            }
        ],
    }


class ModelAndHelperTests(unittest.TestCase):
    def test_friend_note_migration_adds_missing_sections_idempotently(self) -> None:
        note = {
            "target_type": "friend",
            "record_type": "note",
            "content": "Current events:\n- Started a new job",
        }

        migrated = migrate_friend_note(note)

        self.assertIsNotNone(migrated)
        for section in FRIEND_NOTE_SECTIONS:
            self.assertIn(section, migrated)
        note["content"] = migrated
        note["schema_version"] = FRIEND_NOTE_SCHEMA_VERSION
        self.assertIsNone(migrate_friend_note(note))

    def test_friend_note_migration_skips_history_and_projects(self) -> None:
        for note in (
            {"target_type": "friend", "record_type": "history", "content": "raw"},
            {"target_type": "project", "record_type": "note", "content": "work"},
        ):
            self.assertIsNone(migrate_friend_note(note))

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
                "migrate",
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

    def test_proposal_reason_accepts_500_characters_but_not_more(self) -> None:
        payload = proposal_payload()["items"][0].copy()  # type: ignore[index,union-attr]
        payload["reason"] = "r" * 500
        self.assertEqual(len(ProposalItem.model_validate(payload).reason), 500)

        payload["reason"] = "r" * 501
        with self.assertRaises(ValidationError):
            ProposalItem.model_validate(payload)

    def test_follow_up_requires_a_scheduled_time(self) -> None:
        payload = proposal_payload()["items"][0].copy()  # type: ignore[index,union-attr]
        payload["category"] = "follow_up"
        with self.assertRaises(ValidationError):
            ProposalItem.model_validate(payload)

    def test_friend_follow_up_tag_is_rejected(self) -> None:
        payload = proposal_payload()
        payload["items"][0].update(  # type: ignore[index]
            category="follow_up",
            follow_up_at="2026-07-26T09:00:00+08:00",
        )
        with self.assertRaises(ValidationError):
            NoteProposal.model_validate(payload)

    def test_project_follow_up_tag_is_rejected(self) -> None:
        payload = proposal_payload()
        payload["items"][0].update(  # type: ignore[index]
            target_type="project",
            target_name="Portfolio",
            category="follow_up",
            follow_up_at="2026-07-26T09:00:00+08:00",
        )
        with self.assertRaises(ValidationError):
            NoteProposal.model_validate(payload)

    def test_standalone_follow_up_does_not_require_friend_or_project_note(self) -> None:
        payload = proposal_payload()
        payload["items"][0].update(  # type: ignore[index]
            target_type="follow_up",
            target_name="Call Alice",
            category="follow_up",
            follow_up_at="2026-07-26T09:00:00+08:00",
        )

        proposal = NoteProposal.model_validate(payload)

        self.assertEqual(proposal.items[0].target_type, "follow_up")
        self.assertEqual(proposal.items[0].target_name, "Call Alice")

    def test_standalone_follow_up_materializes_without_entity_context(self) -> None:
        payload = operation_payload()
        payload["items"][0].update(  # type: ignore[index]
            target_type="follow_up",
            target_name="Call Alice",
            category="follow_up",
            operations=[],
            content="Call Alice about her application.",
            follow_up_at="2026-07-26T09:00:00+08:00",
        )

        proposal = DeepSeekClassifier._materialize_operations(
            OperationProposal.model_validate(payload),
            [],
        )

        self.assertEqual(proposal.items[0].target_type, "follow_up")
        self.assertEqual(proposal.items[0].content, "Call Alice about her application.")

    def test_low_confidence_standalone_follow_up_moves_to_inbox(self) -> None:
        item = ProposalItem(
            target_type="follow_up",
            target_name="Call Alice",
            category="follow_up",
            content="Call Alice about her application.",
            occurred_on=None,
            follow_up_at="2026-07-26T09:00:00+08:00",
            birthday_mm_dd=None,
            confidence=0.5,
            reason="Explicit reminder.",
        )

        result = DeepSeekClassifier._apply_confidence_threshold(
            NoteProposal(summary="Schedule reminder", items=[item]),
            0.65,
        )

        self.assertEqual(result.items[0].target_type, "uncategorized")
        self.assertEqual(result.items[0].category, "note")
        self.assertIsNone(result.items[0].follow_up_at)

    def test_note_operations_append_merge_replace_and_delete(self) -> None:
        prior = [
            {
                "target_type": "friend",
                "target_name": "Alice",
                "notes": [
                    {
                        "category": "note",
                        "content": (
                            "Current events:\n- Applying to GovTech\n\n"
                            "Upcoming events:\n\n"
                            "Hobbies/interests:\n- Arknights — plays regularly\n\n"
                            "Siblings:\n\nBirthday:\n\nLikes:\n- tea\n\n"
                            "Dislikes:\n\nRelationship with family:"
                        ),
                    }
                ],
            }
        ]
        payload = operation_payload(
            "merge",
            section="Hobbies/interests",
            match="Arknights",
            content="Arknights — plays regularly and follows upcoming events",
        )
        operations = payload["items"][0]["operations"]  # type: ignore[index]
        operations.extend(  # type: ignore[union-attr]
            [
                {
                    "action": "append",
                    "section": "Likes",
                    "match": None,
                    "content": "donuts",
                    "source_quote": "likes donuts",
                    "reason": "New preference.",
                },
                {
                    "action": "replace",
                    "section": "Current events",
                    "match": "Applying to GovTech",
                    "content": "Rejected by GovTech on July 24, 2026",
                    "source_quote": "rejected by GovTech",
                    "reason": "State changed.",
                },
                {
                    "action": "delete",
                    "section": "Likes",
                    "match": "tea",
                    "content": None,
                    "source_quote": "remove tea",
                    "reason": "Explicit removal.",
                },
            ]
        )

        proposal = DeepSeekClassifier._materialize_operations(
            OperationProposal.model_validate(payload),
            prior,
        )

        content = proposal.items[0].content
        self.assertIn(
            "• Arknights — plays regularly and follows upcoming events", content
        )
        self.assertIn("• donuts", content)
        self.assertIn("• Rejected by GovTech on July 24, 2026", content)
        self.assertNotIn("- tea", content)
        self.assertNotIn("Applying to GovTech", content)

    def test_operation_content_removes_model_supplied_bullets(self) -> None:
        for supplied in (
            "• Conducted a Friendfolio test",
            "- Conducted a Friendfolio test",
            "• • Conducted a Friendfolio test",
        ):
            operation = NoteEditOperation(
                action="append",
                section=None,
                match=None,
                content=supplied,
                source_quote="conducted a test",
                reason="New project detail.",
            )

            result = DeepSeekClassifier._apply_operations(
                "",
                "project",
                [operation],
            )

            self.assertEqual(result, "• Conducted a Friendfolio test")

    def test_delete_requires_a_match(self) -> None:
        with self.assertRaises(ValidationError):
            NoteEditOperation.model_validate(
                {
                    "action": "delete",
                    "section": "Likes",
                    "match": None,
                    "content": None,
                    "source_quote": "remove it",
                    "reason": "Explicit removal.",
                }
            )

    def test_project_delete_can_match_a_multiline_block(self) -> None:
        removable = (
            "Deepseek still makes a lot of mistakes. Ask Codex for suggestions.\n\n"
            "Also prefer adding a new row unless needing to merge things."
        )
        current = f"Keep this first paragraph.\n\n{removable}"
        operation = NoteEditOperation(
            action="delete",
            section=None,
            match=removable,
            content=None,
            source_quote=f"remove `{removable}`",
            reason="Explicit removal.",
        )

        result = DeepSeekClassifier._apply_operations(
            current,
            "project",
            [operation],
        )

        self.assertEqual(result, "Keep this first paragraph.")

    def test_operation_match_error_reports_missing_and_ambiguous_matches(self) -> None:
        for current, expected_count in (
            ("something else", 0),
            ("duplicate\nduplicate", 2),
        ):
            operation = NoteEditOperation(
                action="delete",
                section=None,
                match="duplicate",
                content=None,
                source_quote="remove duplicate",
                reason="Explicit removal.",
            )

            with self.assertRaises(NoteOperationError) as caught:
                DeepSeekClassifier._apply_operations(
                    current,
                    "project",
                    [operation],
                )

            self.assertEqual(caught.exception.action, "delete")
            self.assertEqual(caught.exception.match_count, expected_count)

    def test_missing_friend_sections_are_reported_without_changing_content(self) -> None:
        payload = proposal_payload()
        proposal = NoteProposal.model_validate(payload)

        self.assertEqual(
            DeepSeekClassifier.missing_note_sections(proposal)["Alice"],
            list(DeepSeekClassifier.NOTE_SECTIONS),
        )
        self.assertEqual(proposal.items[0].content, "Alice started a new role.")

    def test_deepseek_request_uses_configured_model_and_json_mode(self) -> None:
        captured: list[dict[str, object]] = []

        def responder(request: httpx.Request) -> httpx.Response:
            captured.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": json.dumps(operation_payload())}}
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
        self.assertIn("• Alice started a new role.", result.items[0].content)
        self.assertIn("Relationship with family:", result.items[0].content)
        payload = json.loads(captured[0]["messages"][1]["content"])  # type: ignore[index]
        self.assertEqual(
            payload["prior_context"][0]["follow_up_at"], "2026-07-24T13:00:00+00:00"
        )
        system_prompt = captured[0]["messages"][0]["content"]  # type: ignore[index]
        self.assertIn("small edit operations", system_prompt)
        self.assertIn("append-only history", system_prompt)
        self.assertIn("Operation content is the row text only", system_prompt)
        self.assertIn("application adds the note bullet", system_prompt)
        self.assertIn("Use delete only", system_prompt)
        self.assertIn("reason under 250 characters", system_prompt)
        self.assertIn("standalone target_type=follow_up", system_prompt)
        self.assertIn("clearly named in new_note is a friend target", system_prompt)
        self.assertIn(
            "create a new friend note rather than update an existing friend note",
            system_prompt,
        )
        self.assertIn("relative time words", system_prompt)
        self.assertIn("age N as of YYYY-MM-DD", system_prompt)
        self.assertEqual(len(captured), 2)
        verification_prompt = captured[1]["messages"][0]["content"]  # type: ignore[index]
        self.assertIn("audit proposed note edit operations", verification_prompt)
        self.assertEqual(captured[0]["model"], "deepseek-v4-flash")
        self.assertEqual(captured[0]["response_format"], {"type": "json_object"})
        self.assertEqual(captured[0]["thinking"], {"type": "disabled"})

    def test_deepseek_migration_moves_known_fact_into_new_section(self) -> None:
        sections = "\n\n".join(
            (
                "Current events:",
                "Upcoming events:",
                "Hobbies/interests:",
                "Siblings:",
                "Birthday:",
                "Likes:",
                "Dislikes:",
                "Relationship with family:",
                "Lives at:\n• Singapore",
            )
        )

        def responder(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            self.assertIn("move it there", payload["messages"][0]["content"])
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "items": [
                                            {
                                                "note_id": "note-1",
                                                "content": sections,
                                                "reason": "Moved Singapore into Lives at.",
                                            }
                                        ]
                                    }
                                )
                            }
                        }
                    ]
                },
            )

        classifier = DeepSeekClassifier("test-key", "test-model", "Asia/Singapore")
        classifier.client = httpx.Client(
            transport=httpx.MockTransport(responder)
        )

        result = classifier.migrate_friend_notes(
            [
                {
                    "note_id": "note-1",
                    "friend_name": "Alice",
                    "content": "Current events:\n- Lives in Singapore",
                }
            ],
            ["Lives at:"],
        )

        self.assertIn("Lives at:\n• Singapore", result.items[0].content)

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
        payload = operation_payload()
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
