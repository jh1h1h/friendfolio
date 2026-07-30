from __future__ import annotations

import re
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .note_schema import NOTE_BULLET


TargetType = Literal["friend", "project", "follow_up", "uncategorized"]
Category = Literal["note", "follow_up"]
EditAction = Literal["append", "merge", "replace", "delete"]


class NoteEditOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: EditAction
    section: str | None = Field(default=None, max_length=80)
    match: str | None = Field(default=None, max_length=1000)
    content: str | None = Field(default=None, max_length=3000)
    source_quote: str = Field(min_length=1, max_length=1000)
    reason: str = Field(min_length=1, max_length=500)

    @field_validator("section", "match")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @field_validator("content")
    @classmethod
    def normalize_operation_content(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        prefixes = (f"{NOTE_BULLET} ", "- ")
        while stripped.startswith(prefixes):
            stripped = stripped[2:].lstrip()
        return stripped or None

    @field_validator("source_quote", "reason")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def validate_action_fields(self) -> "NoteEditOperation":
        if self.action == "append" and not self.content:
            raise ValueError("append requires content")
        if self.action in {"merge", "replace"} and (
            not self.match or not self.content
        ):
            raise ValueError(f"{self.action} requires match and content")
        if self.action == "delete" and not self.match:
            raise ValueError("delete requires match")
        return self


class OperationProposalItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_type: TargetType
    target_name: str = Field(max_length=120)
    category: Category
    operations: list[NoteEditOperation] = Field(default_factory=list, max_length=20)
    content: str | None = Field(default=None, max_length=3000)
    occurred_on: date | None
    follow_up_at: datetime | None
    birthday_mm_dd: str | None
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_operation_item(self) -> "OperationProposalItem":
        if self.target_type == "uncategorized":
            self.target_name = ""
            self.category = "note"
            self.birthday_mm_dd = None
        elif not self.target_name.strip():
            raise ValueError("target_name is required")
        if self.birthday_mm_dd:
            ProposalItem.valid_birthday(self.birthday_mm_dd)
            if self.target_type != "friend":
                raise ValueError("birthdays can only be assigned to friends")
        if self.follow_up_at is not None:
            self.category = "follow_up"
        if self.category == "follow_up":
            if self.follow_up_at is None or not self.content:
                raise ValueError("follow_up requires content and follow_up_at")
            if self.operations:
                raise ValueError("follow_up cannot contain note operations")
            if self.target_type != "follow_up":
                raise ValueError("follow_up items must use standalone target_type=follow_up")
        elif self.target_type == "follow_up":
            raise ValueError("target_type=follow_up requires category=follow_up")
        elif not self.operations:
            raise ValueError("note items require at least one operation")
        return self


class OperationProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=300)
    items: list[OperationProposalItem] = Field(min_length=1, max_length=8)


class ProposalItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_type: TargetType
    target_name: str = Field(max_length=120)
    category: Category
    content: str = Field(min_length=1, max_length=20_000)
    occurred_on: date | None
    follow_up_at: datetime | None
    birthday_mm_dd: str | None
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1, max_length=500)

    @field_validator("target_name", "content", "reason")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("birthday_mm_dd")
    @classmethod
    def valid_birthday(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not re.fullmatch(r"\d{2}-\d{2}", value):
            raise ValueError("birthday_mm_dd must use MM-DD")
        month, day = (int(part) for part in value.split("-"))
        date(2024, month, day)
        return value

    @model_validator(mode="after")
    def validate_target(self) -> "ProposalItem":
        if self.target_type == "uncategorized":
            self.target_name = ""
            self.category = "note"
            self.birthday_mm_dd = None
        elif not self.target_name:
            raise ValueError("target_name is required")
        if self.birthday_mm_dd and self.target_type != "friend":
            raise ValueError("birthdays can only be assigned to friends")
        if self.follow_up_at is not None:
            self.category = "follow_up"
        elif self.category == "follow_up":
            raise ValueError("follow_up items require follow_up_at")
        if self.category == "follow_up" and self.target_type != "follow_up":
            raise ValueError("follow_up items must use standalone target_type=follow_up")
        if self.target_type == "follow_up" and self.category != "follow_up":
            raise ValueError("target_type=follow_up requires category=follow_up")
        return self


class NoteProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=300)
    items: list[ProposalItem] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def require_current_note(self) -> "NoteProposal":
        entity_targets = {
            (item.target_type, item.target_name)
            for item in self.items
            if item.target_type in {"friend", "project"}
        }
        note_targets = {
            (item.target_type, item.target_name)
            for item in self.items
            if item.target_type in {"friend", "project"} and item.category == "note"
        }
        if entity_targets - note_targets:
            raise ValueError("every affected friend or project requires an updated note item")
        return self


class ContextSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    friend_names: list[str] = Field(default_factory=list, max_length=8)
    project_names: list[str] = Field(default_factory=list, max_length=8)

    @field_validator("friend_names", "project_names")
    @classmethod
    def strip_names(cls, values: list[str]) -> list[str]:
        return [value.strip() for value in values if value and value.strip()]


class SearchPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=300)
    include_terms: list[str] = Field(default_factory=list, max_length=12)
    exclude_terms: list[str] = Field(default_factory=list, max_length=8)
    entity_names: list[str] = Field(default_factory=list, max_length=8)
    target_types: list[TargetType] = Field(default_factory=list, max_length=4)
    categories: list[Category] = Field(default_factory=list, max_length=6)
    limit: int = Field(default=20, ge=1, le=20)
    sort_by: Literal["relevance", "newest"] = "relevance"
    require_all_terms: bool = False

    @field_validator("include_terms", "exclude_terms", "entity_names")
    @classmethod
    def strip_terms(cls, values: list[str]) -> list[str]:
        return [value.strip() for value in values if value and value.strip()]


class SearchAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1, max_length=3500)


class FriendNoteMigrationItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    note_id: str = Field(min_length=1, max_length=120)
    content: str = Field(min_length=1, max_length=20_000)
    reason: str = Field(min_length=1, max_length=500)


class FriendNoteMigrationProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[FriendNoteMigrationItem] = Field(max_length=8)
