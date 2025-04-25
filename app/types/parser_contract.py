"""Pydantic models that define the contract between the Planner/Parser agent
and the rest of the backend.

These classes are intentionally framework-agnostic so they can be reused by
workers, API responses, and tests without pulling in FastAPI or database
layers.
"""

from __future__ import annotations

from typing import List, Literal, Optional
from datetime import datetime

from pydantic import BaseModel, Field, field_validator, model_validator

Intent = Literal["save", "reminder", "both", "unknown"]


class EntityDraft(BaseModel):
    """A rough entity extracted by the LLM that may still need resolution.

    For example, a place mentioned in text may only include a name and city.
    The resolver worker can enrich it with a lat/lon and external IDs later.
    """

    type: str  # e.g. "place", "event", "tag"
    name: str
    city: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    needs_resolution: bool = True

    # Normalise tags to lowercase for consistency
    @field_validator("tags")
    def _lowercase(cls, v: list[str]):  # noqa: N805
        return [t.lower() for t in v]


# ──────────────────────────────
# MVP: Reminders-only reply
# ──────────────────────────────


class ReminderReply(BaseModel):
    """LLM response for reminders-only MVP.

    Either provides a fully-specified reminder task or asks for clarification.
    """

    need_clarification: bool = False
    clarification_question: Optional[str] = None
    reminder: Optional[ReminderTask] = None

    @model_validator(mode="after")
    def _cross_field_checks(self):  # noqa: N805
        """Ensure logical consistency between fields.

        • If clarification is required, there must be a non-empty clarification_question
        • If no clarification is required, a reminder object must be supplied
        """
        if self.need_clarification:
            if not self.clarification_question or not self.clarification_question.strip():
                raise ValueError("clarification_question must be provided when need_clarification is True")
        else:
            if self.reminder is None:
                raise ValueError("reminder must be provided when no clarification is needed")
        return self


class ReminderTask(BaseModel):
    """The contract for a time-based reminder task, as required by the reminder worker."""
    user_id: str
    reminder_text: str
    reminder_time: datetime  # Parsed datetime; ISO8601 strings are accepted and auto-parsed by Pydantic
    timezone: str       # Olson timezone string, e.g. 'America/New_York'
    channel: str = "sms"  # Delivery channel, default to 'sms'

    @field_validator("timezone")
    def validate_timezone(cls, v):
        try:
            from zoneinfo import ZoneInfo
            ZoneInfo(v)
        except Exception:
            raise ValueError(f"timezone '{v}' is not a valid Olson timezone string")
        return v

    @field_validator("user_id")
    def validate_user_id(cls, v):
        if not isinstance(v, str) or not v.strip():
            raise ValueError("user_id must be a non-empty string")
        return v

    @field_validator("channel")
    def validate_channel(cls, v):
        if v != "sms":
            raise ValueError("channel must be 'sms'")
        return v

    @field_validator("reminder_time")
    def validate_reminder_time(cls, v: datetime):
        if not isinstance(v, datetime):
            raise ValueError("reminder_time must be a datetime instance")
        return v
