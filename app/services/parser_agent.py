"""
LLM-powered Parser/Planner agent.

Converts an SMS/MMS “Envelope” + optional OCR text into a ParserReply.

Author: <you>
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, TypedDict

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
)

from app.types.parser_contract import ParserReply

# ──────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────

_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
_MODEL          = os.getenv("OPENAI_MODEL", "o4-mini")
_TIMEOUT        = float(os.getenv("OPENAI_TIMEOUT", "30"))

client = AsyncOpenAI(api_key=_OPENAI_API_KEY)

_LOGGER = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────
# Prompts & function-tool definition
# ──────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a planning assistant that converts user SMS/MMS into structured "
    "JSON with fields: intent, confidence (0-1), need_clarification, "
    "clarification_question, and entities (list of drafts). "
    "Use the schema provided and THINK step-by-step before responding. "
    "Return ONLY JSON."
)

TEXT_TEMPLATE = (
    "# Envelope\n{envelope}\n\n"
    "# OCR_Text\n{ocr}\n\n"
    "Respond with JSON per the schema."
)

FUNCTION_DEF = {
    "name": "parse_sms",
    "description": "Extract structured intent and entities from an SMS/MMS envelope.",
    "parameters": {
        "type": "object",
        "properties": {
            "intent": {"type": "string", "description": "User intent label"},
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
            "need_clarification": {"type": "boolean"},
            "clarification_question": {"type": "string"},
            "entities": {"type": "array", "items": {"type": "object"}},
        },
        "required": ["intent", "confidence"],
    },
}

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
        tool_choice="auto",
        temperature=0.2,
        timeout=_TIMEOUT,
    )

    tool_calls = response.choices[0].message.tool_calls
    if not tool_calls:
        raise ValueError("LLM did not invoke the parse_sms tool")

    args = tool_calls[0].function.arguments
    # openai-python returns str; but if it ever returns dict, handle gracefully
    return args if isinstance(args, str) else json.dumps(args)


# ──────────────────────────────────────────────────────────────────────────
# Public entry-point
# ──────────────────────────────────────────────────────────────────────────

async def run(envelope: Dict[str, Any], ocr_text: str | None = None) -> ParserReply:
    """
    Parse an MMS/SMS envelope + optional OCR into a structured ParserReply.

    Raises:
        ValueError: if LLM output cannot be parsed into ParserReply
    """

    if not _OPENAI_API_KEY:
        # Fallback stub for local dev without API key
        return ParserReply(intent="save", confidence=0.9, entities=[])

    messages = _build_messages(envelope, ocr_text)

    try:
        raw_json = await _call_openai(messages)
        return ParserReply.parse_raw(raw_json)
    except Exception as exc:
        _LOGGER.exception("Failed to parse LLM JSON: %s", exc)
        raise ValueError(f"Failed to parse LLM JSON: {exc!s}\nRAW: {raw_json}") from exc


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