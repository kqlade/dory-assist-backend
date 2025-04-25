"""LLM-powered Parser/Planner agent.

Given an Envelope node (dict) and optional OCR text, returns a ParserReply
describing the user's intent and extracted entities.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential

from app.types.parser_contract import ParserReply

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "o4-mini")

openai.api_key = OPENAI_API_KEY

SYSTEM_PROMPT = (
    "You are a planning assistant that converts user SMS/MMS into structured "
    "JSON with fields: intent, confidence (0-1), need_clarification, "
    "clarification_question, and entities (list of drafts). Use the schema "
    "provided and THINK step-by-step before responding. Return ONLY JSON."
)

TEXT_TEMPLATE = (
    "# Envelope\n{envelope}\n\n# OCR_Text\n{ocr}\n\n"
    "Respond with JSON per the schema."
)

@retry(wait=wait_random_exponential(multiplier=1, max=30), stop=stop_after_attempt(3))
async def _call_openai(messages: list[dict[str, str]]) -> str:
    response = await openai.ChatCompletion.acreate(
        model=MODEL,
        messages=messages,
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


async def run(envelope: Dict[str, Any], ocr_text: str | None = None) -> ParserReply:
    """Call the LLM and return a validated ParserReply object."""

    if not OPENAI_API_KEY:
        # Fallback stub for local dev without API key
        return ParserReply(intent="save", confidence=0.9, entities=[])

    text_part = TEXT_TEMPLATE.format(envelope=envelope, ocr=ocr_text or "")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # First the text request
    user_msg: dict[str, Any] = {"role": "user", "content": []}
    user_msg["content"].append({"type": "text", "text": text_part})

    # Append each image as separate part
    for img in envelope.get("payload", {}).get("images", []):
        url = img.get("external_url") or img.get("url")
        if url:
            user_msg["content"].append({
                "type": "image_url",
                "image_url": {"url": url}
            })

    messages.append(user_msg)

    raw_json = await _call_openai(messages)
    try:
        return ParserReply.parse_raw(raw_json)
    except Exception as e:
        # Attach raw json for debugging
        raise ValueError(f"Failed to parse LLM JSON: {e}\nRAW: {raw_json}") from e
