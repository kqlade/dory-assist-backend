"""Simple URL→plain-text helper used by fetch_url() tool."""

from __future__ import annotations
import json, openai, os
from openai import AsyncOpenAI
from config import settings

SEARCH_MODEL = settings.OPENAI_SEARCH_MODEL  # likely gpt-4o-search-preview
client       = AsyncOpenAI()

async def _chat_json(prompt: str) -> str:
    """Internal helper: force JSON-object output and return raw assistant text."""
    resp = await client.chat.completions.create(
        model=SEARCH_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "Return ONLY plain text, no markdown.",
            },
            {"role": "user", "content": prompt},
        ],
        timeout=settings.OPENAI_TIMEOUT,
    )
    return resp.choices[0].message.content or ""

async def fetch_url_content(url: str, max_chars: int = 10_000) -> str:
    """Return human-readable text from *url* (truncated)."""
    prompt = f"Fetch {url} and return the visible text (limit {max_chars})."
    try:
        text = await _chat_json(prompt)
    except openai.OpenAIError:
        return ""
    return text[:max_chars]