from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .note_schema import FRIEND_NOTE_SCHEMA_VERSION, FRIEND_NOTE_SECTIONS


FRIEND_NOTES_V1 = "friend-notes-v1"


@dataclass(frozen=True)
class MigrationDefinition:
    migration_id: str
    description: str
    target_version: int


MIGRATIONS = {
    FRIEND_NOTES_V1: MigrationDefinition(
        migration_id=FRIEND_NOTES_V1,
        description="Add every current friend-note section, including Lives at",
        target_version=FRIEND_NOTE_SCHEMA_VERSION,
    )
}


def missing_friend_note_sections(note: dict[str, Any]) -> list[str] | None:
    if note.get("archived_at") is not None or note.get("target_type") != "friend":
        return None
    record_type = note.get("record_type")
    if record_type not in {None, "note", "summary"}:
        return None
    if note.get("category") in {"follow_up", "next_action"}:
        return None
    version = note.get("schema_version", 0)
    if isinstance(version, int) and version >= FRIEND_NOTE_SCHEMA_VERSION:
        return None

    content = str(note.get("content", "")).strip()
    existing = {
        line.strip().casefold()
        for line in content.splitlines()
        if line.strip().endswith(":")
    }
    return [
        heading for heading in FRIEND_NOTE_SECTIONS if heading.casefold() not in existing
    ]


def migrate_friend_note(note: dict[str, Any]) -> str | None:
    missing = missing_friend_note_sections(note)
    if missing is None:
        return None
    content = str(note.get("content", "")).strip()
    additions = "\n\n".join(missing)
    return "\n\n".join(part for part in (content, additions) if part)
