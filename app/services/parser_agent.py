"""
LLM-powered Parser/Planner agent.

Converts an SMS/MMS "Envelope" + optional OCR text into a ParserReply (or the
more specific ReminderReply when operating in reminders-only mode).

Author: <you>
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, TypedDict
from datetime import datetime, timedelta, timezone

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
)

from app.types.parser_contract import ReminderReply, ReminderTask

# ──────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────

_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Use a model that officially supports function/tool calling. Allow override via env.
# As of 2024, o4-mini supports tool/function calling (see OpenAI/third-party docs).
_MODEL          = os.getenv("OPENAI_MODEL", "o4-mini")
_TIMEOUT        = float(os.getenv("OPENAI_TIMEOUT", "30"))

client = AsyncOpenAI(api_key=_OPENAI_API_KEY)

_LOGGER = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────
# Prompts & function-tool definition
# ──────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an intake assistant that extracts time-based reminders from user "
    "SMS/MMS (text + images + links). Use ALL information in the envelope, "
    "including image content, to infer the reminder details. "
    "Return ONLY JSON that matches the provided schema. "
    "If you can fully determine the reminder, set need_clarification=false and "
    "fill the reminder object. Otherwise set need_clarification=true and provide "
    "a single clarification_question asking for the missing info."
)

TEXT_TEMPLATE = (
    "# Envelope\n{envelope}\n\n"
    "# OCR_Text\n{ocr}\n\n"
    "Respond with JSON per the schema."
)

FUNCTION_DEF = {
    "name": "parse_reminder",
    "description": "Extract a ReminderTask or clarification from an SMS/MMS envelope.",
    "parameters": {
        "type": "object",
        "properties": {
            "need_clarification": {"type": "boolean"},
            "clarification_question": {"type": "string"},
            "reminder": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "reminder_text": {"type": "string"},
                    "reminder_time": {"type": "string"},
                    "timezone": {"type": "string"},
                    "channel": {"type": "string", "enum": ["sms"]},
                },
                "required": ["user_id", "reminder_text", "reminder_time", "timezone"],
            },
        },
        "required": ["need_clarification"],
        "additionalProperties": False,
    },
}

INTENT_SYNONYMS: dict[str, str] = {}


def _normalize_llm_json(raw: str) -> str:
    """Fix common schema drift issues before Pydantic validation."""
    try:
        data = json.loads(raw)
    except Exception:
        # If already dict-like, just cast
        if isinstance(raw, dict):
            data = raw
        else:
            raise

    # No special normalization needed for reminders-only MVP
    return json.dumps(data)

# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


class Envelope(TypedDict, total=False):
    """Rough structure of the incoming message envelope."""
    payload: Dict[str, Any]


def _build_messages(envelope: Envelope, ocr_text: str | None) -> List[ChatCompletionMessageParam]:
    """Compose the multimodal message list for Chat Completions."""
    user_content: List[Dict[str, Any]] = [
        {"type": "text", "text": TEXT_TEMPLATE.format(envelope=envelope, ocr=ocr_text or "")}
    ]

    for img in envelope.get("payload", {}).get("images", []):
        url = img.get("external_url") or img.get("url")
        if url:
            user_content.append({"type": "image_url", "image_url": {"url": url}})

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# Retry only on transport / rate-limit / backend errors
RETRY_ERRORS = (
    openai.APIStatusError,
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.APITimeoutError,
)


@retry(
    wait=wait_random_exponential(multiplier=1, max=30),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(RETRY_ERRORS),
)
async def _call_openai(messages: List[ChatCompletionMessageParam]) -> str:
    """Low-level OpenAI call, returns the raw JSON string from tool arguments."""

    response = await client.chat.completions.create(
        model=_MODEL,
        messages=messages,
        tools=[{"type": "function", "function": FUNCTION_DEF}],
        # Force the assistant to call the parse_sms function; if it cannot, the
        # OpenAI API will raise an error instead of silently returning text.
        tool_choice={"type": "function", "function": {"name": "parse_sms"}},
        # temperature=0.2,
        timeout=_TIMEOUT,
    )

    tool_calls = response.choices[0].message.tool_calls

    if tool_calls:
        args = tool_calls[0].function.arguments
        # openai-python returns str; but if it ever returns dict, handle gracefully
        return args if isinstance(args, str) else json.dumps(args)

    # ──────────────────────────────────────────────
    # Fallback: assistant replied with raw JSON text
    # (This should not happen with the enforced tool_choice, but we keep it as
    # an additional guardrail for forward compatibility.)
    # ──────────────────────────────────────────────
    content = response.choices[0].message.content or ""
    try:
        # Validate by round-tripping through json
        json.loads(content)
        return content  # type: ignore[return-value]
    except Exception:
        raise ValueError("LLM did not invoke the parse_sms tool and fallback JSON parse failed")


# ──────────────────────────────────────────────────────────────────────────
# Public entry-point
# ──────────────────────────────────────────────────────────────────────────

async def run(envelope: Dict[str, Any], ocr_text: str | None = None) -> ReminderReply:
    """
    Parse an MMS/SMS envelope + optional OCR into a structured ReminderReply.

    Raises:
        ValueError: if LLM output cannot be parsed into ReminderReply
    """

    # Check for OpenAI credentials at call-time (tests may manipulate env)
    if not os.getenv("OPENAI_API_KEY"):
        # When no OpenAI key (local dev), create a dummy reminder 1 hour in the future
        reminder_dt = datetime.now(tz=timezone.utc) + timedelta(hours=1)
        dummy_reminder = ReminderTask(
            user_id=envelope.get("user_id", "unknown"),
            reminder_text=envelope.get("instruction", "todo"),
            reminder_time=reminder_dt,
            timezone="UTC",
        )

        return ReminderReply(
            need_clarification=False,
            reminder=dummy_reminder,
        )

    messages = _build_messages(envelope, ocr_text)

    raw_json = ""
    try:
        raw_json = await _call_openai(messages)
        return ReminderReply.model_validate_json(_normalize_llm_json(raw_json))
    except Exception as e:
        # Attach raw json for debugging, if any
        msg = f"Failed to get/parse LLM JSON: {e}"
        if raw_json:
            msg += f"\nRAW: {raw_json}"
        raise ValueError(msg) from e


# ──────────────────────────────────────────────────────────────────────────
# Optional: quick CLI test
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import asyncio, sys, pprint, json as _json

    dummy_env = {
        "payload": {
            "images": [
                {"url": "https://example.com/foo.jpg"},
            ]
        }
    }
    ocr = "Buy 2 tickets for Friday at 7 pm"

    pprint.pp(asyncio.run(run(dummy_env, ocr)))