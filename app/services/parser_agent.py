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

from app.types.parser_contract import ReminderReply, ReminderTask, TimeTrigger

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
                    "triggers": {"type": "array", "items": {"type": "object"}},
                    "channel": {"type": "string", "enum": ["sms"]},
                },
                "required": ["user_id", "reminder_text", "triggers"],
            },
        },
        "required": ["need_clarification"],
        "additionalProperties": False,
    },
}

INTENT_SYNONYMS: dict[str, str] = {}

FUNCTIONS = [
    {"type": "function", "function": FUNCTION_DEF},
    {
        "type": "function",
        "function": {
            "name": "lookup_reminders",
            "description": "Search past reminders for the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "keyword": {"type": "string"},
                    "limit": {"type": "integer", "default": 5},
                },
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_envelopes",
            "description": "Search past message envelopes for the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "keyword": {"type": "string"},
                    "limit": {"type": "integer", "default": 5},
                },
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url_content",
            "description": "Download a URL and return cleaned markdown for the LLM.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "max_chars": {"type": "integer", "default": 10000},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_photo_metadata",
            "description": "Return EXIF datetime, GPS and camera details for an image URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                },
                "required": ["url"],
            },
        },
    },
]

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
        tool_choice={"type": "function", "function": {"name": "parse_reminder"}},
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


async def _run_openai_with_tools(messages: List[ChatCompletionMessageParam]) -> str:
    """Loop calling OpenAI, executing tools until we get parse_reminder."""
    max_iters = 3
    for _ in range(max_iters):
        response = await client.chat.completions.create(
            model=_MODEL,
            messages=messages,
            tools=FUNCTIONS,
            tool_choice={"type": "function", "function": "parse_reminder"},
            timeout=_TIMEOUT,
        )
        msg = response.choices[0].message
        if msg.tool_calls:
            call = msg.tool_calls[0]
            name = call.function.name
            args = json.loads(call.function.arguments)
            # Execute tool
            if name == "lookup_reminders":
                from db import search_reminders  # local import to avoid cycles
                results = await search_reminders(
                    user_id=args["user_id"],
                    keyword=args.get("keyword"),
                    limit=args.get("limit", 5),
                )
                payload = json.dumps(results)
            elif name == "lookup_envelopes":
                from db import search_envelopes
                results = await search_envelopes(
                    user_id=args["user_id"],
                    keyword=args.get("keyword"),
                    limit=args.get("limit", 5),
                )
                payload = json.dumps(results)
            elif name == "parse_reminder":
                return call.function.arguments  # already JSON str
            elif name == "fetch_url_content":
                from app.utils.web_fetch import fetch_url_content
                payload = await fetch_url_content(args["url"], args.get("max_chars", 10000))
            elif name == "fetch_photo_metadata":
                from app.utils.photo_metadata import extract_photo_metadata
                meta = await asyncio.to_thread(extract_photo_metadata, args["url"])
                payload = json.dumps(meta)
            else:
                payload = "{}"
            # Append tool response and loop
            messages.append({"role": "tool", "tool_call_id": call.id, "name": name, "content": payload})
        else:
            # No tool call; assume assistant responded with final JSON
            return msg.content or ""
    raise ValueError("Exceeded max tool iterations without parse_reminder")


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
        dummy_trigger = TimeTrigger(at=reminder_dt, timezone="UTC")
        dummy_reminder = ReminderTask(
            user_id=envelope.get("user_id", "unknown"),
            reminder_text=envelope.get("instruction", "todo"),
            triggers=[dummy_trigger],
        )

        return ReminderReply(
            need_clarification=False,
            reminder=dummy_reminder,
        )

    messages = _build_messages(envelope, ocr_text)

    raw_json = ""
    try:
        raw_json = await _run_openai_with_tools(messages)
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