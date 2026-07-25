from __future__ import annotations

import re
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


TargetType = Literal["friend", "project", "uncategorized"]
Category = Literal["note", "follow_up"]


class ProposalItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_type: TargetType
    target_name: str = Field(max_length=120)
    category: Category
    content: str = Field(min_length=1, max_length=3000)
    occurred_on: date | None
    follow_up_at: datetime | None
    birthday_mm_dd: str | None
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1, max_length=300)

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
            raise ValueError("target_name is required for a friend or project")
        if self.birthday_mm_dd and self.target_type != "friend":
            raise ValueError("birthdays can only be assigned to friends")
        if self.follow_up_at is not None:
            self.category = "follow_up"
        elif self.category == "follow_up":
            raise ValueError("follow_up items require follow_up_at")
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
    target_types: list[TargetType] = Field(default_factory=list, max_length=3)
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
