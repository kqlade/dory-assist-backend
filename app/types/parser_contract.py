"""Pydantic models that define the contract between the Planner/Parser agent
and the rest of the backend.

These classes are intentionally framework-agnostic so they can be reused by
workers, API responses, and tests without pulling in FastAPI or database
layers.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

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
    @field_validator("tags", each_item=True)
    def _lowercase(cls, v: str) -> str:  # noqa: N805
        return v.lower()


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

    @field_validator("clarification_question")
    def _clarification_present_if_needed(cls, v, values):  # noqa: N805
        if values.get("need_clarification") and (v is None or not v.strip()):
            raise ValueError("clarification_question must be provided when need_clarification is True")
        return v

    @field_validator("reminder")
    def _reminder_present_if_no_clarification(cls, v, values):  # noqa: N805
        if not values.get("need_clarification") and v is None:
            raise ValueError("reminder must be provided when no clarification is needed")
        return v


class ReminderTask(BaseModel):
    """The contract for a time-based reminder task, as required by the reminder worker."""
    user_id: str
    reminder_text: str
    reminder_time: str  # ISO8601 timestamp, e.g. '2024-06-28T09:00:00'
    timezone: str       # Olson timezone string, e.g. 'America/New_York'
    channel: str = "sms"  # Delivery channel, default to 'sms'

    @field_validator("reminder_time")
    def validate_reminder_time(cls, v):
        from datetime import datetime
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError("reminder_time must be a valid ISO8601 timestamp")
        return v

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
