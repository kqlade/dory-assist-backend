"""Pydantic models that define the contract between the Planner/Parser agent
and the rest of the backend.

These classes are intentionally framework-agnostic so they can be reused by
workers, API responses, and tests without pulling in FastAPI or database
layers.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple, Union
from typing_extensions import Annotated
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


class TimeTrigger(BaseModel):
    """Trigger based on a specific moment in time."""

    type: Literal["time"] = "time"
    at: datetime
    timezone: str

    @field_validator("timezone")
    def _validate_tz(cls, v):  # noqa: N805
        try:
            from zoneinfo import ZoneInfo
            ZoneInfo(v)
        except Exception:
            raise ValueError(f"timezone '{v}' is not a valid Olson timezone string")
        return v


class LocationPoint(BaseModel):
    lat: float
    lon: float


class LocationTrigger(BaseModel):
    """Trigger that fires when the user is within `radius_m` of *any* point."""

    type: Literal["location"] = "location"
    points: List[LocationPoint]
    radius_m: int = 100  # default 100 m geofence
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None


class EventTrigger(BaseModel):
    """Trigger relative to a calendar event by id + offset."""

    type: Literal["event"] = "event"
    calendar_event_id: str
    offset_minutes: int  # negative → before, positive → after (0 = at)


Trigger = Annotated[Union[TimeTrigger, LocationTrigger, EventTrigger], Field(discriminator="type")]


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
    """The contract for a reminder task, supporting multiple trigger types."""

    user_id: str
    reminder_text: str
    triggers: List[Trigger]
    channel: str = "sms"  # Delivery channel, default to 'sms'

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @field_validator("user_id")
    def validate_user_id(cls, v):  # noqa: N805
        if not isinstance(v, str) or not v.strip():
            raise ValueError("user_id must be a non-empty string")
        return v

    @field_validator("channel")
    def validate_channel(cls, v):  # noqa: N805
        if v != "sms":
            raise ValueError("channel must be 'sms'")
        return v

    @model_validator(mode="after")
    def _ensure_trigger(cls, model):  # noqa: N805
        """Ensure at least one trigger is provided."""
        if not getattr(model, "triggers", None):
            raise ValueError("at least one trigger must be provided")
        return model
