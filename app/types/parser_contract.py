"""Pydantic models that define the contract between the Planner/Parser agent
and the rest of the backend.

These classes are intentionally framework-agnostic so they can be reused by
workers, API responses, and tests without pulling in FastAPI or database
layers.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, validator

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
    @validator("tags", each_item=True)
    def _lowercase(cls, v: str) -> str:  # noqa: N805
        return v.lower()


class ParserReply(BaseModel):
    """The structured response returned by the Planner/Parser agent."""

    intent: Intent = "unknown"
    confidence: float = Field(ge=0, le=1)
    need_clarification: bool = False
    clarification_question: Optional[str] = None
    entities: List[EntityDraft] = Field(default_factory=list)

    @validator("confidence")
    def _clamp_confidence(cls, v: float) -> float:  # noqa: N805
        # Guard against occasional float rounding errors from the LLM.
        return max(0.0, min(1.0, v))
