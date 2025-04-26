"""
LLM-powered Parser/Planner agent.

Converts an SMS/MMS *Envelope* into a `ReminderReply`.

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
import db

# ──────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Use a model that officially supports function/tool calling. Allow override via env.
# As of 2024, o4-mini supports tool/function calling (see OpenAI/third-party docs).
OPENAI_MODEL          = os.getenv("OPENAI_MODEL", "o4-mini")
OPENAI_TIMEOUT        = float(os.getenv("OPENAI_TIMEOUT", "30"))

_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

_LOGGER = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────
# Prompts & function-tool definition
# ──────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are an intake assistant that extracts **time-based reminders** from "
    "user SMS/MMS (text, images and links). "
    "Return ONLY JSON that matches the given schema. "
    "If you can fully determine the reminder, create the `reminder` object. "
    "**Crucially:** Every `reminder` *must* include a `triggers` array with at least one valid trigger. "
    "If you cannot determine a specific trigger (or any required field), you *must* set "
    "`need_clarification=true` and provide exactly one `clarification_question` asking for the missing info. "
    "Do NOT return a `reminder` object if clarification is required."
    "\n\n"
    "# Example – clarification needed\n"
    "User says: Remind me to pay rent\n"
    "Assistant JSON:\n"
    "{\n  \"need_clarification\": true,\n  \"clarification_question\": \"Sure – when should I remind you to pay rent?\"\n}\n\n"
    "# Example – complete reminder\n"
    "User says: Remind me tomorrow at 9am to pay rent\n"
    "Assistant JSON:\n"
    "{\n  \"need_clarification\": false,\n  \"reminder\": {\n    \"user_id\": \"5551234\",\n    \"reminder_text\": \"Pay rent\",\n    \"triggers\": [\n      {\n        \"type\": \"time\",\n        \"at\": \"2025-05-01T09:00:00Z\",\n        \"timezone\": \"UTC\"\n      }\n    ],\n    \"channel\": \"sms\"\n  }\n}\n"
)

_TEXT_TEMPLATE = (
    "# Envelope\n{envelope}\n\n"
    "# OCR_Text\n{ocr}\n\n"
    "Respond with JSON per the schema."
)

_FUNCTION_DEF = {
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

INTENT_SYNONYMS: dict[str, str] = {}

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
    """Truncate JSON serialized object to max_chars."""
    txt = json.dumps(obj, ensure_ascii=False)
    return txt if len(txt) <= max_chars else txt[: max_chars] + " …truncated…"

# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


class Envelope(TypedDict, total=False):
    """Rough structure of the incoming message envelope."""
    payload: Dict[str, Any]


def _build_messages(env: Dict[str, Any], ocr_text: str | None = None) -> List[ChatCompletionMessageParam]:
    """Compose a multimodal message list for Chat Completions."""
    pruned_env = {
        "from": env.get("user_id"),
        "body": env.get("body", "")[:280] if "body" in env else env.get("instruction", "")[:280],
        "images": [img.get("url") or img.get("external_url") for img in env.get("images", [])],
        "timestamp": env.get("timestamp"),
    }
    user_payload: List[Dict[str, Any]] = [
        {"type": "text", "text": _TEXT_TEMPLATE.format(
            envelope=_safe_truncate(pruned_env), ocr=""
        )}
    ]
    for url in pruned_env["images"]:
        if url:
            user_payload.append({"type": "image_url", "image_url": {"url": url}})

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_payload},
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
    """Call OpenAI with the parse_reminder tool and return JSON string."""
    _LOGGER.debug("Calling OpenAI with parse_reminder tool")
    rsp = await _client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        tools=[{"type": "function", "function": _FUNCTION_DEF}],
        tool_choice={"type": "function", "function": {"name": "parse_reminder"}},
        timeout=OPENAI_TIMEOUT,
    )
    msg = rsp.choices[0].message
    
    # Get JSON either from tool arguments or message content
    if msg.tool_calls:
        result = msg.tool_calls[0].function.arguments
        _LOGGER.debug(f"Tool call result: {result[:100]}...")
        return result
    
    # Fallback to content if no tool call (shouldn't happen with tool_choice specified)
    result = msg.content or ""
    _LOGGER.debug(f"Content result (no tool call): {result[:100]}...")
    return result


async def _run_openai_with_tools(messages: List[ChatCompletionMessageParam]) -> str:
    """Loop calling OpenAI, executing tools until we get parse_reminder."""
    max_iters = 3
    for _ in range(max_iters):
        response = await _client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            tools=FUNCTIONS,
            tool_choice={"type": "function", "function": {"name": "parse_reminder"}},
            timeout=OPENAI_TIMEOUT,
        )
        msg = response.choices[0].message
        if msg.tool_calls:
            call = msg.tool_calls[0]
            name = call.function.name
            args = json.loads(call.function.arguments)
            # Execute tool
            if name == "lookup_reminders":
                results = await db.search_reminders(
                    user_id=args["user_id"],
                    keyword=args.get("keyword"),
                    limit=args.get("limit", 5),
                )
                payload = json.dumps(results)
            elif name == "lookup_envelopes":
                results = await db.search_envelopes(
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

async def run(envelope: Dict[str, Any]) -> ReminderReply:
    """
    Parse an MMS/SMS envelope into a structured `ReminderReply`.

    Falls back to a dummy "one-off in 1 h" reminder if no OpenAI key present.
    """
    if not OPENAI_API_KEY:
        # local dev shortcut
        dt = datetime.now(timezone.utc) + timedelta(hours=1)
        return ReminderReply(
            need_clarification=False,
            reminder=ReminderTask(
                user_id=envelope.get("user_id", "unknown"),
                reminder_text=envelope.get("body", envelope.get("instruction", "todo")),
                triggers=[TimeTrigger(at=dt, timezone="UTC")],
            ),
        )

    try:
        msgs = _build_messages(envelope)
        raw_json = await _call_openai(msgs)
        
        # Dump raw LLM output for debugging
        _LOGGER.debug(f"Raw LLM output: {raw_json[:500]}")
        
        # Validate by parsing then re-serializing to ensure proper JSON
        parsed_data = json.loads(raw_json) if isinstance(raw_json, str) else raw_json
        _LOGGER.debug(f"Parsed data: {json.dumps(parsed_data, indent=2)}")
        
        # Add more defensive validation before Pydantic validation
        if isinstance(parsed_data, dict):
            # Handle missing triggers or empty triggers array
            if "reminder" in parsed_data and isinstance(parsed_data["reminder"], dict):
                reminder = parsed_data["reminder"]
                
                # Check if triggers is missing completely
                if "triggers" not in reminder:
                    _LOGGER.error(f"LLM returned reminder without triggers field: {json.dumps(reminder)}")
                    # Force it to need clarification instead
                    parsed_data["need_clarification"] = True
                    parsed_data["clarification_question"] = "When should I remind you about this?"
                    parsed_data.pop("reminder", None)  # Remove invalid reminder
                
                # Check if triggers array is empty
                elif not reminder.get("triggers"):
                    _LOGGER.error(f"LLM returned reminder with empty triggers array: {json.dumps(reminder)}")
                    # Force it to need clarification instead
                    parsed_data["need_clarification"] = True
                    parsed_data["clarification_question"] = "When should I remind you about this?"
                    parsed_data.pop("reminder", None)  # Remove invalid reminder
                
                # Check if triggers array contains objects without 'type' field
                elif isinstance(reminder["triggers"], list) and any(
                    not isinstance(t, dict) or "type" not in t 
                    for t in reminder["triggers"]
                ):
                    _LOGGER.error(f"LLM returned invalid trigger objects: {json.dumps(reminder['triggers'])}")
                    # Force it to need clarification instead
                    parsed_data["need_clarification"] = True
                    parsed_data["clarification_question"] = "I couldn't understand when to remind you. Please specify a time or date."
                    parsed_data.pop("reminder", None)  # Remove invalid reminder
        
        normalized_json = json.dumps(parsed_data, ensure_ascii=False)
        
        return ReminderReply.model_validate_json(normalized_json)
    except Exception as exc:
        _LOGGER.warning(f"Failed to parse assistant output: {exc}, raw: {raw_json[:200]}")
        raise ValueError(f"Unable to interpret LLM output: {exc}") from exc


# ──────────────────────────────────────────────────────────────────────────
# Optional: quick CLI test
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import asyncio, sys, pprint, json as _json

    dummy_env = {
        "user_id": "example_user",
        "instruction": "Remind me to buy 2 tickets for Friday at 7 pm",
        "payload": {
            "images": [
                {"url": "https://example.com/foo.jpg"},
            ]
        }
    }
    
    pprint.pp(asyncio.run(run(dummy_env)))

# ─────────────────────────── background helper ─────────────────────── #

async def run_and_store(envelope: Dict[str, Any]):
    """
    Run the parser and store the reminder if found.
    """
    try:
        _LOGGER.info(f"Processing envelope: {envelope.get('envelope_id')} from user {envelope.get('user_id')}")
        reply = await run(envelope)
        
        if reply.need_clarification:
            _LOGGER.info(f"Clarification needed: {reply.clarification_question}")
            # Could send SMS here requesting clarification
            return
            
        if not reply.reminder:
            _LOGGER.warning("No reminder in reply and no clarification requested")
            return
            
        # Extra validation for the triggers
        if not reply.reminder.triggers:
            _LOGGER.error("Empty triggers list in reminder (should have been caught earlier)")
            return
            
        # Log what kind of trigger we're using
        trigger_types = [t.type for t in reply.reminder.triggers]
        _LOGGER.info(f"Storing reminder with trigger types: {trigger_types}")
        
        # Store reminder
        await db.insert_reminder(reply.reminder)
        _LOGGER.info(f"Successfully stored reminder for user {reply.reminder.user_id}")
    except Exception as exc:
        _LOGGER.error(f"run_and_store failed: {exc}")
        # For detailed debugging:
        if envelope and isinstance(envelope, dict):
            _LOGGER.error(f"Envelope that caused error: id={envelope.get('envelope_id')}, user={envelope.get('user_id')}")
            _LOGGER.error(f"Instruction: {envelope.get('instruction', '')[:100]}")