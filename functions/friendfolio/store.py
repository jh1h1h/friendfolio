from __future__ import annotations

import hashlib
import re
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any, Sequence
from zoneinfo import ZoneInfo

from firebase_admin import firestore

from .models import NoteProposal, SearchPlan


UTC = timezone.utc
DEFAULT_CONFIDENCE_THRESHOLD = 0.65


def utc_now() -> datetime:
    return datetime.now(UTC)


def normalize_name(name: str) -> str:
    return " ".join(name.strip().casefold().split())


def entity_id(name: str) -> str:
    return hashlib.sha256(normalize_name(name).encode()).hexdigest()[:24]


def safe_id(*parts: object) -> str:
    raw = ":".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def as_utc(value: datetime | None, local_timezone: str) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=ZoneInfo(local_timezone))
    return value.astimezone(UTC)


def _sort_time(value: Any) -> datetime:
    return value if isinstance(value, datetime) else datetime.min.replace(tzinfo=UTC)


class FirestoreRegistry:
    def __init__(self, client: Any, local_timezone: str = "Asia/Singapore") -> None:
        self.client = client
        self.local_timezone = local_timezone

    def user_ref(self, owner_user_id: int) -> Any:
        return self.client.collection("users").document(str(owner_user_id))

    def collection(self, owner_user_id: int, name: str) -> Any:
        return self.user_ref(owner_user_id).collection(name)

    @staticmethod
    def _entity_collection(target_type: str) -> str:
        if target_type == "friend":
            return "friends"
        if target_type == "project":
            return "projects"
        raise ValueError("target_type must be friend or project")

    def ensure_user(self, owner_user_id: int) -> None:
        self.user_ref(owner_user_id).set(
            {
                "telegram_user_id": owner_user_id,
                "updated_at": firestore.SERVER_TIMESTAMP,
            },
            merge=True,
        )

    def get_confidence_threshold(self, owner_user_id: int) -> float:
        snapshot = self.user_ref(owner_user_id).get()
        if not snapshot.exists:
            return DEFAULT_CONFIDENCE_THRESHOLD
        value = (snapshot.to_dict() or {}).get(
            "confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD
        )
        if isinstance(value, (int, float)) and 0 <= float(value) <= 1:
            return float(value)
        return DEFAULT_CONFIDENCE_THRESHOLD

    def set_confidence_threshold(self, owner_user_id: int, value: float) -> None:
        if not 0 <= value <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1")
        self.user_ref(owner_user_id).set(
            {
                "telegram_user_id": owner_user_id,
                "confidence_threshold": value,
                "updated_at": firestore.SERVER_TIMESTAMP,
            },
            merge=True,
        )

    def upsert_entity(
        self, owner_user_id: int, target_type: str, name: str
    ) -> tuple[str, bool]:
        clean_name = " ".join(name.strip().split())
        if not clean_name or len(clean_name) > 120:
            raise ValueError("Name must contain 1 to 120 characters")
        collection = self._entity_collection(target_type)
        ref = self.collection(owner_user_id, collection).document(entity_id(clean_name))
        created = not ref.get().exists
        ref.set(
            {
                "name": clean_name,
                "normalized_name": normalize_name(clean_name),
                "updated_at": firestore.SERVER_TIMESTAMP,
                **(
                    {"created_at": firestore.SERVER_TIMESTAMP, "note_count": 0}
                    if created
                    else {}
                ),
            },
            merge=True,
        )
        self.ensure_user(owner_user_id)
        return ref.id, created

    def list_entities(
        self, owner_user_id: int, target_type: str
    ) -> list[dict[str, Any]]:
        collection = self._entity_collection(target_type)
        rows = []
        for snapshot in self.collection(owner_user_id, collection).stream():
            item = snapshot.to_dict() or {}
            item["id"] = snapshot.id
            rows.append(item)
        return sorted(rows, key=lambda item: str(item.get("name", "")).casefold())

    def list_entity_names(self, owner_user_id: int, target_type: str) -> list[str]:
        return [
            str(row["name"]) for row in self.list_entities(owner_user_id, target_type)
        ]

    def all_notes(
        self, owner_user_id: int, include_archived: bool = False
    ) -> list[dict[str, Any]]:
        rows = []
        for snapshot in self.collection(owner_user_id, "notes").stream():
            item = snapshot.to_dict() or {}
            if not include_archived and item.get("archived_at") is not None:
                continue
            item["id"] = snapshot.id
            rows.append(item)
        return rows

    def resolve_note_id(self, owner_user_id: int, value: str) -> str | None:
        matches = [
            note["id"]
            for note in self.all_notes(owner_user_id)
            if note["id"].startswith(value)
        ]
        return matches[0] if len(matches) == 1 else None

    def get_entity_notes(
        self, owner_user_id: int, target_type: str, name: str, limit: int = 25
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        collection = self._entity_collection(target_type)
        target_id = entity_id(name)
        snapshot = self.collection(owner_user_id, collection).document(target_id).get()
        if not snapshot.exists:
            return None, []
        entity = snapshot.to_dict() or {}
        entity["id"] = snapshot.id
        notes = [
            note
            for note in self.all_notes(owner_user_id)
            if note.get("target_type") == target_type
            and note.get("target_id") == target_id
        ]
        notes.sort(
            key=lambda note: _sort_time(
                note.get("updated_at") or note.get("created_at")
            ),
            reverse=True,
        )
        return entity, notes[:limit]

    def stage_pending(
        self,
        owner_user_id: int,
        raw_input: str,
        proposal: NoteProposal,
        expiry_hours: int,
        token: str | None = None,
        source_note_id: str | None = None,
        debug_mode: bool = False,
    ) -> tuple[str, str]:
        token = token or uuid.uuid4().hex
        ref = self.collection(owner_user_id, "pending_actions").document(token)
        transaction = self.client.transaction()

        @firestore.transactional
        def create_if_absent(txn: Any) -> str:
            existing = ref.get(transaction=txn)
            if existing.exists:
                return str((existing.to_dict() or {}).get("status", "pending"))
            now = utc_now()
            txn.set(
                ref,
                {
                    "raw_input": raw_input,
                    "proposal": proposal.model_dump(mode="json"),
                    "source_note_id": source_note_id,
                    "debug_mode": debug_mode,
                    "status": "pending",
                    "created_at": now,
                    "expires_at": now + timedelta(hours=expiry_hours),
                },
            )
            return "pending"

        status = create_if_absent(transaction)
        self.ensure_user(owner_user_id)
        return token, status

    def latest_pending_action(self, owner_user_id: int) -> dict[str, Any] | None:
        rows: list[dict[str, Any]] = []
        for snapshot in self.collection(owner_user_id, "pending_actions").stream():
            data = snapshot.to_dict() or {}
            if str(data.get("status", "missing")) != "pending":
                continue
            data["id"] = snapshot.id
            rows.append(data)
        rows.sort(key=lambda item: _sort_time(item.get("created_at")), reverse=True)
        return rows[0] if rows else None

    def set_pending_message(
        self, owner_user_id: int, token: str, chat_id: int, message_id: int
    ) -> bool:
        ref = self.collection(owner_user_id, "pending_actions").document(token)
        snapshot = ref.get()
        if not snapshot.exists:
            return False
        data = snapshot.to_dict() or {}
        if str(data.get("status", "missing")) != "pending":
            return False
        ref.update({"chat_id": chat_id, "message_id": message_id})
        return True

    def revise_pending(
        self,
        owner_user_id: int,
        token: str,
        proposal: NoteProposal,
        instruction: str,
    ) -> str:
        ref = self.collection(owner_user_id, "pending_actions").document(token)
        transaction = self.client.transaction()

        @firestore.transactional
        def revise(txn: Any) -> str:
            snapshot = ref.get(transaction=txn)
            if not snapshot.exists:
                return "missing"
            data = snapshot.to_dict() or {}
            status = str(data.get("status", "missing"))
            if status != "pending":
                return status
            if data.get("expires_at") and data["expires_at"] <= utc_now():
                txn.update(ref, {"status": "expired", "decided_at": utc_now()})
                return "expired"
            txn.update(
                ref,
                {
                    "proposal": proposal.model_dump(mode="json"),
                    "last_instruction": instruction,
                    "revised_at": utc_now(),
                },
            )
            return "pending"

        return revise(transaction)

    def cancel_pending(self, owner_user_id: int, token: str) -> str:
        ref = self.collection(owner_user_id, "pending_actions").document(token)
        transaction = self.client.transaction()

        @firestore.transactional
        def cancel(txn: Any) -> str:
            snapshot = ref.get(transaction=txn)
            if not snapshot.exists:
                return "missing"
            data = snapshot.to_dict() or {}
            status = str(data.get("status", "missing"))
            if status != "pending":
                return status
            if data.get("expires_at") and data["expires_at"] <= utc_now():
                txn.update(ref, {"status": "expired", "decided_at": utc_now()})
                return "expired"
            txn.update(ref, {"status": "cancelled", "decided_at": utc_now()})
            return "cancelled"

        return cancel(transaction)

    def approve_pending(
        self, owner_user_id: int, token: str
    ) -> tuple[str, list[dict[str, str]]]:
        pending_ref = self.collection(owner_user_id, "pending_actions").document(token)
        transaction = self.client.transaction()

        @firestore.transactional
        def approve(txn: Any) -> tuple[str, list[dict[str, str]]]:
            snapshot = pending_ref.get(transaction=txn)
            if not snapshot.exists:
                return "missing", []
            data = snapshot.to_dict() or {}
            status = str(data.get("status", "missing"))
            if status != "pending":
                return status, []
            if data.get("expires_at") and data["expires_at"] <= utc_now():
                txn.update(pending_ref, {"status": "expired", "decided_at": utc_now()})
                return "expired", []

            proposal = NoteProposal.model_validate(data["proposal"])
            # A friend/project has one consolidated ordinary note, while scheduled
            # follow-ups remain separate. For duplicate items in the same category,
            # the classifier's last item is the update.
            items_by_entity: dict[tuple[str, str, str], Any] = {}
            items_to_save: list[Any] = []
            for item in proposal.items:
                if item.target_type in {"friend", "project"}:
                    key = (
                        item.target_type,
                        entity_id(item.target_name),
                        item.category,
                    )
                    if key not in items_by_entity:
                        items_to_save.append(key)
                    items_by_entity[key] = item
                else:
                    items_to_save.append(item)

            resolved_items = [
                items_by_entity[value] if isinstance(value, tuple) else value
                for value in items_to_save
            ]

            # Firestore transactions require every read to happen before the first
            # write. Resolve existing notes up front so approval can update one in
            # place and archive any duplicates left by older versions.
            existing_notes: dict[tuple[str, str, str], list[Any]] = {}
            notes_collection = self.collection(owner_user_id, "notes")
            for item in resolved_items:
                if item.target_type not in {"friend", "project"}:
                    continue
                target_id = entity_id(item.target_name)
                key = (item.target_type, target_id, item.category)
                if item.category == "follow_up":
                    existing_notes[key] = []
                    continue
                query = notes_collection.where(
                    "target_type", "==", item.target_type
                ).where("target_id", "==", target_id)
                existing_notes[key] = [
                    note
                    for note in query.stream(transaction=txn)
                    if (note.to_dict() or {}).get("archived_at") is None
                    and (note.to_dict() or {}).get("category")
                    not in {"follow_up", "next_action"}
                    and (note.to_dict() or {}).get("record_type") != "history"
                ]

            saved_notes: list[dict[str, str]] = []
            for index, item in enumerate(resolved_items):
                target_id: str | None = None
                active_notes: list[Any] = []
                if item.target_type in {"friend", "project"}:
                    target_id = entity_id(item.target_name)
                    active_notes = existing_notes[
                        (item.target_type, target_id, item.category)
                    ]
                    entity_collection = self._entity_collection(item.target_type)
                    entity_ref = self.collection(
                        owner_user_id, entity_collection
                    ).document(target_id)
                    entity_data: dict[str, Any] = {
                        "name": item.target_name,
                        "normalized_name": normalize_name(item.target_name),
                        "updated_at": firestore.SERVER_TIMESTAMP,
                    }
                    if item.target_type == "project":
                        entity_data["note_count"] = firestore.Increment(
                            1 if not active_notes else -(len(active_notes) - 1)
                        )
                    if item.birthday_mm_dd:
                        entity_data["birthday_mm_dd"] = item.birthday_mm_dd
                    txn.set(entity_ref, entity_data, merge=True)

                if active_notes:
                    active_notes.sort(
                        key=lambda note: _sort_time(
                            (note.to_dict() or {}).get("created_at")
                        ),
                        reverse=True,
                    )
                    note_ref = active_notes[0].reference
                    note_id = note_ref.id
                    for duplicate in active_notes[1:]:
                        txn.update(
                            duplicate.reference,
                            {"archived_at": firestore.SERVER_TIMESTAMP},
                        )
                else:
                    note_id = safe_id(token, index)
                    note_ref = notes_collection.document(note_id)

                follow_up_at = as_utc(item.follow_up_at, self.local_timezone)
                note_data = {
                    "target_type": item.target_type,
                    "target_id": target_id,
                    "target_name": item.target_name or "Inbox",
                    "category": item.category,
                    "record_type": (
                        "follow_up"
                        if item.category == "follow_up"
                        else "note"
                    ),
                    "content": item.content,
                    "raw_input": data["raw_input"],
                    "occurred_on": item.occurred_on.isoformat()
                    if item.occurred_on
                    else None,
                    "follow_up_at": follow_up_at,
                    "follow_up_status": "pending"
                    if item.category == "follow_up"
                    else None,
                    "confidence": item.confidence,
                    "archived_at": None,
                    "updated_at": firestore.SERVER_TIMESTAMP,
                }
                if not active_notes:
                    note_data["created_at"] = firestore.SERVER_TIMESTAMP
                txn.set(note_ref, note_data, merge=bool(active_notes))
                saved_notes.append(
                    {
                        "id": note_id,
                        "action": "updated" if active_notes else "saved",
                        "target_name": item.target_name or "Inbox",
                        "target_type": item.target_type,
                        "record_type": note_data["record_type"],
                        "content": item.content,
                    }
                )

            history_targets = {
                (item.target_type, entity_id(item.target_name), item.target_name)
                for item in resolved_items
                if item.target_type in {"friend", "project"}
            }
            for target_type, target_id, target_name in history_targets:
                history_id = safe_id(token, "history", target_type, target_id)
                history_ref = notes_collection.document(history_id)
                txn.set(
                    history_ref,
                    {
                        "target_type": target_type,
                        "target_id": target_id,
                        "target_name": target_name,
                        "category": "note",
                        "record_type": "history",
                        "content": data["raw_input"],
                        "raw_input": data["raw_input"],
                        "occurred_on": None,
                        "follow_up_at": None,
                        "follow_up_status": None,
                        "archived_at": None,
                        "created_at": firestore.SERVER_TIMESTAMP,
                        "updated_at": firestore.SERVER_TIMESTAMP,
                    },
                )
                entity_ref = self.collection(
                    owner_user_id, self._entity_collection(target_type)
                ).document(target_id)
                txn.set(
                    entity_ref,
                    {
                        "history_note_count": firestore.Increment(1),
                        "updated_at": firestore.SERVER_TIMESTAMP,
                    },
                    merge=True,
                )
                saved_notes.append(
                    {
                        "id": history_id,
                        "action": "saved",
                        "target_name": target_name,
                        "target_type": target_type,
                        "record_type": "history",
                        "content": str(data["raw_input"]),
                    }
                )

            source_note_id = data.get("source_note_id")
            if source_note_id:
                source_ref = self.collection(owner_user_id, "notes").document(
                    source_note_id
                )
                txn.update(source_ref, {"archived_at": firestore.SERVER_TIMESTAMP})
            txn.update(
                pending_ref,
                {"status": "approved", "decided_at": firestore.SERVER_TIMESTAMP},
            )
            return "approved", saved_notes

        return approve(transaction)

    def list_uncategorized(
        self, owner_user_id: int, limit: int = 20
    ) -> list[dict[str, Any]]:
        rows = [
            note
            for note in self.all_notes(owner_user_id)
            if note.get("target_type") == "uncategorized"
        ]
        rows.sort(key=lambda note: _sort_time(note.get("created_at")))
        return rows[:limit]

    def get_uncategorized(
        self, owner_user_id: int, note_id: str
    ) -> dict[str, Any] | None:
        snapshot = self.collection(owner_user_id, "notes").document(note_id).get()
        if not snapshot.exists:
            return None
        item = snapshot.to_dict() or {}
        if (
            item.get("target_type") != "uncategorized"
            or item.get("archived_at") is not None
        ):
            return None
        item["id"] = snapshot.id
        return item

    def pending_followups(
        self, owner_user_id: int, limit: int = 30
    ) -> list[dict[str, Any]]:
        rows = [
            note
            for note in self.all_notes(owner_user_id)
            if note.get("follow_up_status") == "pending"
        ]
        rows.sort(
            key=lambda note: (
                note.get("follow_up_at") is None,
                _sort_time(note.get("follow_up_at")),
            )
        )
        return rows[:limit]

    def due_followups(
        self, owner_user_id: int, now: datetime | None = None
    ) -> list[dict[str, Any]]:
        now = now or utc_now()
        return [
            note
            for note in self.pending_followups(owner_user_id, limit=1000)
            if isinstance(note.get("follow_up_at"), datetime)
            and note["follow_up_at"] <= now
        ]

    def project_next_actions(
        self, owner_user_id: int, limit: int = 30
    ) -> list[dict[str, Any]]:
        return [
            note
            for note in self.pending_followups(owner_user_id, limit=1000)
            if note.get("target_type") == "project"
            and note.get("category") in {"follow_up", "next_action"}
        ][:limit]

    def complete_followup(self, owner_user_id: int, note_id: str) -> bool:
        ref = self.collection(owner_user_id, "notes").document(note_id)
        snapshot = ref.get()
        if (
            not snapshot.exists
            or (snapshot.to_dict() or {}).get("follow_up_status") != "pending"
        ):
            return False
        ref.update(
            {"follow_up_status": "done", "completed_at": firestore.SERVER_TIMESTAMP}
        )
        return True

    def snooze_followup(
        self, owner_user_id: int, note_id: str, until: datetime
    ) -> bool:
        ref = self.collection(owner_user_id, "notes").document(note_id)
        snapshot = ref.get()
        if (
            not snapshot.exists
            or (snapshot.to_dict() or {}).get("follow_up_status") != "pending"
        ):
            return False
        ref.update({"follow_up_at": until.astimezone(UTC)})
        return True

    @staticmethod
    def _normalize_search_text(value: str) -> str:
        return " ".join(value.casefold().split())

    @staticmethod
    def _tokenize_search_text(value: str) -> list[str]:
        return re.findall(r"[\w']+", value.casefold())

    def search_notes(
        self,
        owner_user_id: int,
        query: str,
        limit: int = 20,
        plan: SearchPlan | None = None,
    ) -> list[dict[str, Any]]:
        search_plan = plan or SearchPlan(
            summary="Fallback search plan",
            include_terms=self._tokenize_search_text(query),
            limit=limit,
        )
        include_terms = [self._normalize_search_text(term) for term in search_plan.include_terms]
        exclude_terms = [self._normalize_search_text(term) for term in search_plan.exclude_terms]
        entity_names = [self._normalize_search_text(name) for name in search_plan.entity_names]
        target_types = set(search_plan.target_types)
        categories = set(search_plan.categories)
        query_text = self._normalize_search_text(query)

        rows: list[tuple[int, datetime, dict[str, Any]]] = []
        for note in self.all_notes(owner_user_id):
            note_text = self._normalize_search_text(
                " ".join(
                    str(part)
                    for part in [
                        note.get("content", ""),
                        note.get("raw_input", ""),
                        note.get("target_name", ""),
                        note.get("category", ""),
                        note.get("target_type", ""),
                    ]
                )
            )
            if target_types and note.get("target_type") not in target_types:
                continue
            if categories:
                note_category = str(note.get("category", ""))
                category_matches = note_category in categories or (
                    "note" in categories
                    and note_category
                    in {"general", "status", "like", "dislike", "birthday"}
                )
                if not category_matches:
                    continue
            if exclude_terms and any(term in note_text for term in exclude_terms):
                continue

            score = 0
            if include_terms:
                matched_terms = [term for term in include_terms if term and term in note_text]
                if search_plan.require_all_terms and len(matched_terms) != len(include_terms):
                    continue
                score += len(matched_terms) * 4
            else:
                matched_terms = []

            if query_text and query_text in note_text:
                score += 5
            if entity_names and self._normalize_search_text(str(note.get("target_name", ""))) in entity_names:
                score += 6
            elif entity_names and any(entity in note_text for entity in entity_names):
                score += 3

            if note.get("target_type") == "uncategorized":
                score += 1

            if score <= 0 and include_terms:
                continue
            rows.append((score, _sort_time(note.get("created_at")), note))

        if search_plan.sort_by == "newest":
            rows.sort(key=lambda item: (item[1], item[0]), reverse=True)
        else:
            rows.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [note for _, _, note in rows[: search_plan.limit or limit]]

    @staticmethod
    def _birthday_for_year(mm_dd: str, year: int) -> date:
        month, day = (int(part) for part in mm_dd.split("-"))
        try:
            return date(year, month, day)
        except ValueError:
            return date(year, 2, 28)

    def birthdays_at_offsets(
        self, owner_user_id: int, today: date, offsets: Sequence[int]
    ) -> list[dict[str, Any]]:
        wanted = set(offsets)
        results = []
        for friend in self.list_entities(owner_user_id, "friend"):
            mm_dd = friend.get("birthday_mm_dd")
            if not mm_dd:
                continue
            birthday = self._birthday_for_year(mm_dd, today.year)
            if birthday < today:
                birthday = self._birthday_for_year(mm_dd, today.year + 1)
            days_until = (birthday - today).days
            if days_until in wanted:
                friend["days_until"] = days_until
                friend["next_birthday"] = birthday.isoformat()
                results.append(friend)
        return sorted(
            results, key=lambda item: (item["days_until"], item["name"].casefold())
        )

    def has_notification(
        self, owner_user_id: int, kind: str, reference_key: str, local_date: date
    ) -> bool:
        ref = self.collection(owner_user_id, "notification_log").document(
            safe_id(kind, reference_key, local_date.isoformat())
        )
        return ref.get().exists

    def mark_notification(
        self, owner_user_id: int, kind: str, reference_key: str, local_date: date
    ) -> None:
        ref = self.collection(owner_user_id, "notification_log").document(
            safe_id(kind, reference_key, local_date.isoformat())
        )
        ref.set(
            {
                "kind": kind,
                "reference_key": reference_key,
                "local_date": local_date.isoformat(),
                "sent_at": firestore.SERVER_TIMESTAMP,
            }
        )
