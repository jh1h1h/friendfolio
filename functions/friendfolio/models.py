from __future__ import annotations

import re
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


TargetType = Literal["friend", "project", "uncategorized"]
Category = Literal[
    "general",
    "status",
    "like",
    "dislike",
    "birthday",
    "follow_up",
    "next_action",
]


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
            self.category = "general"
            self.birthday_mm_dd = None
        elif not self.target_name:
            raise ValueError("target_name is required for a friend or project")
        if self.target_type == "project" and self.category in {
            "like",
            "dislike",
            "birthday",
        }:
            raise ValueError("project category is not valid")
        if self.birthday_mm_dd and self.target_type != "friend":
            raise ValueError("birthdays can only be assigned to friends")
        return self


class NoteProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=300)
    items: list[ProposalItem] = Field(min_length=1, max_length=8)


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
