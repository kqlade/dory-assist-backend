"""
LLM-powered Parser/Planner agent.

Converts an SMS/MMS *Envelope* into a `ReminderReply`.

Author: <you>
Synopsis: from app.services.parser_agent import run
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, TypedDict
import datetime as dt

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
import db as db_io

# ──────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────

from typing import Final

try:
    OPENAI_API_KEY: Final = os.environ["OPENAI_API_KEY"].strip()
    OPENAI_MODEL: Final = os.environ["OPENAI_MODEL"].strip()
    OPENAI_TIMEOUT: Final = float(os.getenv("OPENAI_TIMEOUT", "30"))
except KeyError as e:
    raise RuntimeError(f"Missing env var: {e}") from None
except ValueError as e:
    raise RuntimeError(f"Invalid config value: {e}") from None

MODEL_SUPPORTS_TEMP: Final = not OPENAI_MODEL.startswith(("o3", "o4"))

if not MODEL_SUPPORTS_TEMP and "OPENAI_TEMPERATURE" in os.environ:
    raise RuntimeError(
        f"{OPENAI_MODEL} ignores temperature/top_p; "
        "remove OPENAI_TEMPERATURE from the environment."
    )

_client = AsyncOpenAI(api_key=OPENAI_API_KEY, timeout=OPENAI_TIMEOUT)

__all__ = ["run", "run_and_store"]

_LOGGER = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────
# Prompts & function-tool definition
# ──────────────────────────────────────────────────────────────────────────

# TODO: Move prompts to prompts/ directory and load at import time.
_SYSTEM_PROMPT: Final = (
    "You are an intake assistant that extracts **time-based reminders** from "
    "user SMS/MMS (text, images and links).  You have access to the message's "
    "`timestamp` field, which is the ISO-8601 UTC time when the message arrived.  "
    "Whenever the user expresses a relative time (e.g. 'in 5 minutes', 'in 2 hours', 'in 3 days'), convert it into an absolute UTC ISO-8601 timestamp by adding "
    "that offset to the envelope's `timestamp`.  \n\n"
    "Return **only** JSON that matches the given schema.  If you can fully determine "
    "the reminder, create the `reminder` object with a non-empty `triggers` array.  "
    "If you cannot determine a specific trigger (or any required field), set "
    "`need_clarification=true` and provide exactly one `clarification_question`.  "
    "Do **not** invent dates or times if none are given; ask instead.  \n\n"
    "# Example – relative time\n"
    "Envelope.timestamp: \"2025-04-26T00:00:00Z\"\n"
    "User says: Remind me in 5 minutes to stretch\n"
    "Assistant JSON:\n"
    "{\n"
    "  \"need_clarification\": false,\n"
    "  \"reminder\": {\n"
    "    \"user_id\": \"5551234\",\n"
    "    \"reminder_text\": \"stretch\",\n"
    "    \"triggers\": [\n"
    "      {\"type\": \"time\", \"at\": \"2025-04-26T00:05:00Z\", \"timezone\": \"UTC\"}\n"
    "    ],\n"
    "    \"channel\": \"sms\"\n"
    "  }\n"
    "}\n\n"
    "# Example – clarification needed\n"
    "User says: Remind me to call John\n"
    "Assistant JSON:\n"
    "{\n"
    "  \"need_clarification\": true,\n"
    "  \"clarification_question\": \"Sure – what time should I remind you to call John?\"\n"
    "}\n\n"
    "# Example – multiple triggers (recurring)\n"
    "User says: Remind me to take my medicine every day at 9am\n"
    "Assistant JSON:\n"
    "{\n"
    "  \"need_clarification\": false,\n"
    "  \"reminder\": {\n"
    "    \"user_id\": \"5551234\",\n"
    "    \"reminder_text\": \"take my medicine\",\n"
    "    \"triggers\": [\n"
    "      {\"type\": \"time\", \"at\": \"2025-04-27T09:00:00Z\", \"timezone\": \"UTC\"},\n"
    "      {\"type\": \"recurrence\", \"pattern\": \"daily\", \"time\": \"09:00\"}\n"
    "    ],\n"
    "    \"channel\": \"sms\"\n"
    "  }\n"
    "}\n\n"
    "# Example – image in envelope\n"
    "User sends an image of a prescription with the text: 'Remind me to refill in 2 weeks'\n"
    "Assistant JSON:\n"
    "{\n"
    "  \"need_clarification\": false,\n"
    "  \"reminder\": {\n"
    "    \"user_id\": \"5551234\",\n"
    "    \"reminder_text\": \"refill prescription\",\n"
    "    \"triggers\": [\n"
    "      {\"type\": \"time\", \"at\": \"2025-05-10T00:00:00Z\", \"timezone\": \"UTC\"}\n"
    "    ],\n"
    "    \"channel\": \"sms\"\n"
    "  }\n"
    "}\n\n"
    "# Example – non-UTC timezone\n"
    "Envelope.timestamp: \"2025-04-26T08:00:00-07:00\"\n"
    "User says: Remind me at 10am\n"
    "Assistant JSON:\n"
    "{\n"
    "  \"need_clarification\": false,\n"
    "  \"reminder\": {\n"
    "    \"user_id\": \"5551234\",\n"
    "    \"reminder_text\": \"(unspecified)\",\n"
    "    \"triggers\": [\n"
    "      {\"type\": \"time\", \"at\": \"2025-04-26T10:00:00-07:00\", \"timezone\": \"America/Los_Angeles\"}\n"
    "    ],\n"
    "    \"channel\": \"sms\"\n"
    "  }\n"
    "}\n\n"
    "# Example – negative (bad JSON, unknown key)\n"
    "User says: Remind me to water the plants\n"
    "Assistant JSON:\n"
    "{\n"
    "  \"need_clarification\": false,\n"
    "  \"reminder\": {\n"
    "    \"user_id\": \"5551234\",\n"
    "    \"reminder_text\": \"water the plants\",\n"
    "    \"triggers\": [\n"
    "      {\"type\": \"time\", \"at\": \"2025-04-26T18:00:00Z\", \"timezone\": \"UTC\"}\n"
    "    ],\n"
    "    \"channel\": \"sms\",\n"
    "    \"foo\": \"bar\"  # ❌ This is invalid; do not invent unknown keys.\n"
    "  }\n"
    "}\n\n"
    "# Example – ambiguous/unrecognized instruction\n"
    "User says: Just checking in!\n"
    "Assistant JSON:\n"
    "{\n"
    "  \"need_clarification\": true,\n"
    "  \"clarification_question\": \"Could you please specify what you want to be reminded about and when?\"\n"
    "}\n"
)


_TEXT_TEMPLATE: Final = (
    "# Envelope\n{envelope}\n\n"
    "# OCR_Text\n{ocr}\n\n"
    "Respond with JSON per the schema."
)

_FUNCTION_DEF: Final = {
    "name": "parse_reminder",
    "description": "Extract a ReminderTask or ask a single clarification question.",
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
                    "triggers": {
                        "type": "array",
                        "items": {"type": "object"},
                        "minItems": 1  # must include at least one trigger
                    },
                    "channel": {"type": "string", "enum": ["sms"]},
                },
                "required": ["user_id", "reminder_text", "triggers"],
            },
        },
        "required": ["need_clarification"],
        "additionalProperties": False,
    },
}
# INTENT_SYNONYMS removed as dead code

FUNCTIONS = [
    {"type": "function", "function": _FUNCTION_DEF},
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

def _safe_truncate(obj: Any, max_chars: int = 2_000) -> str:
    """Truncate JSON serialized object to max_chars, avoiding mid-codepoint cuts."""
    txt = json.dumps(obj, ensure_ascii=False)
    if len(txt) <= max_chars:
        return txt
    # Avoid cutting in the middle of a unicode code point
    truncated = txt[:max_chars].rstrip("\uFFFD")
    return truncated + " …truncated…"

# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


class Envelope(TypedDict, total=False):
    """Rough structure of the incoming message envelope."""
    payload: Dict[str, Any]


def _build_messages(env: Dict[str, Any]) -> List[ChatCompletionMessageParam]:
    """Compose a multimodal message list for Chat Completions."""
    pruned_env = {
        "from": env.get("user_id"),
        "body": (env.get("body") or env.get("instruction") or "")[:280],
        "images": [img.get("url") or img.get("external_url") for img in env.get("images", [])],
        "timestamp": env.get("timestamp"),
    }
    user_payload: List[Dict[str, Any]] = [
        {"type": "text", "text": _TEXT_TEMPLATE.format(
            envelope=_safe_truncate(pruned_env), ocr=""
        )}
    ]
    # Limit to first 3 images to avoid token bloat
    for url in pruned_env["images"][:3]:
        if url:
            user_payload.append({"type": "image_url", "image_url": {"url": url}})

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_payload},
    ]
    # Add assistant ack if nothing is extractable (empty body/images)
    if not pruned_env["body"] and not pruned_env["images"]:
        messages.append({"role": "assistant", "content": "ack"})
    return messages


# Retry only on transport / rate-limit / backend errors
RETRY_ERRORS = (
    openai.APIStatusError,
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.InternalServerError,  # Added for 5xx
)




# Import tool functions at top to avoid dynamic import overhead
from app.utils.web_fetch import fetch_url_content
from app.utils.photo_metadata import extract_photo_metadata

MAX_TOOL_ITERS: Final = 3

async def _run_openai_with_tools(messages: List[ChatCompletionMessageParam]) -> str:
    """Loop calling OpenAI, executing tools until we get parse_reminder."""
    # Build kwargs for temperature only if supported
    kwargs = {}
    if MODEL_SUPPORTS_TEMP:
        kwargs["temperature"] = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
    for _ in range(MAX_TOOL_ITERS):
        response = await _client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            tools=FUNCTIONS,
            tool_choice={"type": "function", "function": {"name": "parse_reminder"}},
            timeout=OPENAI_TIMEOUT,
            **kwargs,
        )
        msg = response.choices[0].message
        if msg.tool_calls:
            messages.append(msg)
            call = msg.tool_calls[0]
            name = call.function.name
            args = json.loads(call.function.arguments)
            # Map function names to callables
            tool_map = {
                "lookup_reminders": lambda: db_io.search_reminders(
                    user_id=args["user_id"],
                    keyword=args.get("keyword"),
                    limit=args.get("limit", 5),
                ),
                "lookup_envelopes": lambda: db_io.search_envelopes(
                    user_id=args["user_id"],
                    keyword=args.get("keyword"),
                    limit=args.get("limit", 5),
                ),
                "fetch_url_content": lambda: fetch_url_content(args["url"], args.get("max_chars", 10000)),
                "fetch_photo_metadata": lambda: asyncio.to_thread(extract_photo_metadata, args["url"]),
            }
            if name == "parse_reminder":
                data = json.loads(call.function.arguments)
                if not data.get("need_clarification", False):
                    task = data.get("reminder", {})
                    # Guard against malformed triggers
                    triggers = task.get("triggers")
                    if not triggers or not isinstance(triggers, list) or not all(isinstance(t, dict) for t in triggers):
                        clarification_text = f"Sure – when should I remind you about \"{task.get('reminder_text', 'that')}\"?"
                        return json.dumps({
                            "need_clarification": True,
                            "clarification_question": clarification_text
                        })
                return call.function.arguments
            elif name in tool_map:
                result = await tool_map[name]()
                payload = json.dumps(result) if not isinstance(result, str) else result
            else:
                payload = "{}"
            messages.append({"role": "tool", "tool_call_id": call.id, "name": name, "content": payload})
        else:
            _LOGGER.error(f"LLM failed to call required function. Response: {msg}")
            raise ValueError("Assistant failed to call the required function")
    raise ValueError("Exceeded max tool iterations without parse_reminder")


# ──────────────────────────────────────────────────────────────────────────
# Public entry-point
# ──────────────────────────────────────────────────────────────────────────

async def run(envelope: Dict[str, Any]) -> ReminderReply:
    """
    Parse an MMS/SMS envelope into a structured `ReminderReply`.
    Falls back to a dummy "one-off in 1 h" reminder if no OpenAI key present.
    """
    if not OPENAI_API_KEY:
        # local dev shortcut
        dt_obj = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)
        return ReminderReply(
            need_clarification=False,
            reminder=ReminderTask(
                user_id=envelope.get("user_id", "unknown"),
                reminder_text=envelope.get("body", envelope.get("instruction", "todo")),
                triggers=[TimeTrigger(at=dt_obj, timezone="UTC")],
            ),
        )

    try:
        msgs = _build_messages(envelope)
        raw_json = await _run_openai_with_tools(msgs)
        # Validate by parsing then re-serializing to ensure proper JSON
        parsed_data = json.loads(raw_json) if isinstance(raw_json, str) else raw_json
        if isinstance(parsed_data, dict) and "need_clarification" not in parsed_data:
            parsed_data = {"need_clarification": False, "reminder": parsed_data}
        normalized_json = json.dumps(parsed_data, ensure_ascii=False)
        return ReminderReply.model_validate_json(normalized_json)
    except Exception as exc:
        _LOGGER.warning("Failed to parse assistant output: %s", exc)
        raise ValueError("Unable to interpret LLM output") from exc


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

    print("\n--- Result ---")
    pprint.pp(asyncio.run(run(dummy_env)).model_dump())

# ─────────────────────────── background helper ─────────────────────── #

async def run_and_store(envelope: Dict[str, Any]):
    """
    Run the parser and store the reminder if found.
    """
    try:
        reply = await run(envelope)
        if not reply.need_clarification and reply.reminder:
            await db_io.insert_reminder(reply.reminder)
        else:
            # TODO: send_sms(...) for clarification
            pass
    except Exception as exc:
        _LOGGER.error(f"run_and_store failed: {exc}", extra={"user_id": envelope.get("user_id")})