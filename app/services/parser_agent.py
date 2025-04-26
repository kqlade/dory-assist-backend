"""
LLM-powered Parser / Planner  –  OpenAI Agents SDK (2025-04)

 • Reasoning = o-series model via Responses API
 • Live web search = GPT-4o-search-preview + WebSearchTool
 • Tools autowired with @function_tool
 • Output schema unchanged: ReminderReply
"""

from __future__ import annotations

# ───────── std-lib ─────────────────────────────────────────────────────────
import asyncio, json, logging, pathlib, datetime as dt
from typing import Any, Dict, Final, List
from config import settings

# ───────── agents SDK ──────────────────────────────────────────────────────
from agents import (
    Agent,
    Runner,
    function_tool,
    WebSearchTool,
    ModelSettings,
)

# (API key picked up automatically; see docs “Configuring the SDK”)
# ───────── local modules ───────────────────────────────────────────────────
from app.types.parser_contract import ReminderReply, ReminderTask, TimeTrigger
import db as db_io
from app.utils.web_fetch import fetch_url_content
from app.utils.photo_metadata import extract_photo_metadata

# ╔════════════════════════ Config ═════════════════════════════════════════╗
REASON_MODEL: Final = settings.OPENAI_REASON_MODEL
SEARCH_MODEL: Final = settings.OPENAI_SEARCH_MODEL
TIMEOUT_SEC: Final = settings.OPENAI_TIMEOUT

LOG = logging.getLogger(__name__)

# ╔══════════════ prompt assets ══════════════════════════════════════════╗
BASE_DIR = pathlib.Path(settings.AGENT_ASSET_DIR)
_SYSTEM_PROMPT = "\n".join(
    [
        (BASE_DIR / settings.AGENT_PROMPT_FILE).read_text(),
        "\n### Additional Modules\n",
        (BASE_DIR / settings.AGENT_MODULES_FILE).read_text(),
        "\n### Agent Loop\n",
        (BASE_DIR / settings.AGENT_LOOP_FILE).read_text(),
    ]
).strip()

# ╔════════════════════ tools (function_tool) ══════════════════════════════╗
@function_tool()
async def lookup_reminders(user_id: str, keyword: str = "", limit: int = settings.DEFAULT_LOOKUP_LIMIT) -> list[dict]:
    """Return reminders for a user containing a keyword."""
    return await db_io.search_reminders(user_id=user_id, keyword=keyword, limit=limit)


@function_tool()
async def lookup_envelopes(user_id: str, keyword: str = "", limit: int = settings.DEFAULT_LOOKUP_LIMIT) -> list[dict]:
    """Return recent envelopes containing a keyword."""
    return await db_io.search_envelopes(user_id=user_id, keyword=keyword, limit=limit)


@function_tool()
async def fetch_url(url: str, max_chars: int = settings.DEFAULT_MAX_CHARS) -> str:
    """Fetch a URL and return plain-text (truncated)."""
    return await fetch_url_content(url, max_chars)


@function_tool()
async def fetch_photo_metadata_tool(url: str) -> dict:
    """Return EXIF + dimensions for an image."""
    return await asyncio.to_thread(extract_photo_metadata, url)


@function_tool()
async def fetch_search_and_content(
    query: str,
    max_results: int = settings.DEFAULT_MAX_RESULTS,
    max_chars: int = settings.DEFAULT_MAX_CHARS,
) -> str:
    """
    Delegates live search to a GPT-4o-powered agent and returns JSON
    with (url, title, content) tuples.
    """
    search_agent = Agent(
        instructions="Return JSON list of search results.",
        model=SEARCH_MODEL,
        tools=[WebSearchTool()],
    )
    prompt = (
        f"Search the web for {query!r}. "
        f"Give top {max_results} pages with keys url, title, content "
        f"(content ≤ {max_chars} chars). Respond in JSON only."
    )
    run = await Runner.run(search_agent, prompt, timeout_sec=TIMEOUT_SEC)
    return run.final_output  # already JSON string


# ╔══════════════ reasoning agent (o-series) ═══════════════════════════════╗
reason_agent = Agent(
    name="ReminderParser",
    instructions=_SYSTEM_PROMPT,
    model=REASON_MODEL,
    model_settings=ModelSettings(
        temperature=float(os.environ.get("OPENAI_TEMPERATURE", "0.7"))
        if not REASON_MODEL.startswith(("o3-", "o4-"))
        else None,
    ),
    tools=[
        lookup_reminders,
        lookup_envelopes,
        fetch_url,
        fetch_photo_metadata_tool,
        fetch_search_and_content,
    ],
)

# ╔══════════════ helpers to build user input ══════════════════════════════╗
_TMPL = "# Envelope\n{env}\n\nRespond with JSON per schema."


def _safe_trunc(o: Any, n: int = 2_000) -> str:
    t = json.dumps(o, ensure_ascii=False, default=str)
    return t if len(t) <= n else t[: n - 1] + "…"


def _get(e: Dict, k: str, d: Any = None):
    return e.get(k, e.get("payload", {}).get(k, d))


def _build_user_msg(env: Dict[str, Any]) -> str:
    ts = _get(env, "created_at")
    ts = ts.isoformat() if isinstance(ts, dt.datetime) else ts
    summary = {
        "from": _get(env, "user_id"),
        "body": (_get(env, "body") or _get(env, "instruction") or "")[:280],
        "images": [img.get("url") for img in _get(env, "images", [])[:3]],
        "timestamp": ts,
        "timezone": _get(env, "timezone") or settings.DEFAULT_TIMEZONE,
    }
    return _TMPL.format(env=_safe_trunc(summary))


# ╔════════════════ main API ═══════════════════════════════════════════════╗
class ParseFailure(Exception):
    pass


async def run(envelope: Dict[str, Any]) -> ReminderReply:
    """Turn an Envelope → ReminderReply (Agents SDK handles the loop)."""
    if not settings.OPENAI_API_KEY:
        # offline stub
        when = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)
        tz   = envelope.get("timezone") or settings.DEFAULT_TIMEZONE
        return ReminderReply(
            need_clarification=False,
            reminder=ReminderTask(
                user_id=_get(envelope, "user_id", "unknown"),
                reminder_text=_get(envelope, "body", "todo"),
                triggers=[TimeTrigger(at=when, timezone=tz)],
            ),
        )

    user_msg = _build_user_msg(envelope)
    try:
        result = await Runner.run(reason_agent, user_msg, timeout_sec=TIMEOUT_SEC)
        raw    = result.final_output
        data   = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(data, dict) and "need_clarification" not in data:
            data = {"need_clarification": False, "reminder": data}
        return ReminderReply.model_validate_json(json.dumps(data, ensure_ascii=False))
    except Exception as exc:
        LOG.error("Parser failure: %s", exc)
        raise ParseFailure from exc


async def run_and_store(envelope: Dict[str, Any], clarification_handler=None):
    try:
        reply = await run(envelope)
        if not reply.need_clarification and reply.reminder:
            await db_io.insert_reminder(reply.reminder)
        elif clarification_handler:
            await clarification_handler(reply, envelope)
    except Exception as exc:
        LOG.error("run_and_store failed: %s", exc, extra={"user_id": _get(envelope, "user_id")})


# ╔════════════════ smoke-test ═════════════════════════════════════════════╗
if __name__ == "__main__":
    dummy = {
        "payload": {
            "user_id": "5551234",
            "body": "remind me tomorrow at 8 AM to stretch",
            "timestamp": "2025-04-26T00:00:00Z",
            "images": [],
        }
    }
    import pprint, asyncio

    pprint.pp(asyncio.run(run(dummy)).model_dump())