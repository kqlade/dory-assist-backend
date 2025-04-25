import os
import json
import asyncpg

from uuid import uuid4
from datetime import datetime, timezone
from typing import Optional

from app.types.parser_contract import ReminderTask

_pool = None

async def get_pool():
    """Get (or create) a global asyncpg pool."""
    global _pool
    if _pool is None:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise RuntimeError("DATABASE_URL environment variable is not set")
        _pool = await asyncpg.create_pool(database_url, min_size=1, max_size=5)
    return _pool

async def insert_envelope(envelope: dict):
    """Insert an envelope row into message_envelopes table."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO message_envelopes (
                envelope_id,
                user_id,
                channel,
                instruction,
                payload,
                status,
                created_at
            ) VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7)
            ON CONFLICT (envelope_id) DO NOTHING
            """,
            envelope["envelope_id"],
            envelope["user_id"],
            envelope["channel"],
            envelope["instruction"],
            json.dumps(envelope["payload"]),
            envelope.get("status", "received"),
            envelope["created_at"],
        )

# ──────────────────────────────────────────────
# Reminder helpers
# ──────────────────────────────────────────────

async def insert_reminder(task: ReminderTask):
    """Insert a pending reminder row and return its generated UUID.

    For backward compatibility, we retain `reminder_time` and `timezone` columns. If the
    task contains at least one TimeTrigger, we use its `at` and `timezone`. Otherwise we
    store NULL for those columns. The full task is stored in a JSONB `payload` column so
    new trigger types are preserved.
    """

    # Extract first TimeTrigger if present (back-compat)
    reminder_time: Optional[datetime] = None
    timezone: Optional[str] = None
    for trig in task.triggers:
        if isinstance(trig, dict):
            # triggers may come as dict pre-validation; but in runtime they are BaseModel
            ttype = trig.get("type")
        else:
            ttype = getattr(trig, "type", None)

        if ttype == "time":
            if isinstance(trig, dict):
                reminder_time = trig.get("at")
                timezone = trig.get("timezone")
            else:
                reminder_time = trig.at  # type: ignore[attr-defined]
                timezone = trig.timezone  # type: ignore[attr-defined]
            break

    reminder_id = str(uuid4())
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO reminders (
                reminder_id, user_id, reminder_text, reminder_time, timezone, channel,
                payload, status, created_at, updated_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, 'pending', NOW(), NOW())
            """,
            reminder_id,
            task.user_id,
            task.reminder_text,
            reminder_time,
            timezone,
            task.channel,
            json.dumps(task.model_dump(mode="json")),
        )

    return reminder_id

async def fetch_due_reminders(limit: int = 100):
    """Return list of due pending reminders (dicts)."""

    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT * FROM reminders
            WHERE status = 'pending' AND reminder_time <= NOW()
            ORDER BY reminder_time ASC
            LIMIT $1
            """,
            limit,
        )
    return [dict(r) for r in rows]

async def mark_reminder_sent(reminder_id: str):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE reminders
            SET status='sent', updated_at=NOW()
            WHERE reminder_id=$1
            """,
            reminder_id,
        )

async def mark_reminder_failed(reminder_id: str, error: str):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE reminders
            SET status='failed', last_error=$2, updated_at=NOW()
            WHERE reminder_id=$1
            """,
            reminder_id,
            error,
        )

async def create_reminders_table():
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reminders (
                reminder_id   UUID PRIMARY KEY,
                user_id       TEXT NOT NULL,
                reminder_text TEXT NOT NULL,
                reminder_time TIMESTAMPTZ,
                timezone      TEXT,
                channel       TEXT NOT NULL DEFAULT 'sms',
                status        TEXT NOT NULL DEFAULT 'pending',
                last_error    TEXT,
                created_at    TIMESTAMPTZ DEFAULT NOW(),
                updated_at    TIMESTAMPTZ DEFAULT NOW(),
                payload       JSONB
            );
            CREATE INDEX IF NOT EXISTS reminders_due_idx ON reminders (status, reminder_time);
            """
        )

async def create_message_envelopes_table():
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS message_envelopes (
                envelope_id  UUID PRIMARY KEY,
                user_id      TEXT,
                channel      TEXT CHECK (channel IN ('sms','mms','email')),
                instruction  TEXT NOT NULL,
                payload      JSONB,
                raw_refs     JSONB,
                status       TEXT NOT NULL DEFAULT 'received',
                created_at   TIMESTAMPTZ DEFAULT now()
            );
            """
        )

async def search_reminders(user_id: str, keyword: str | None = None, start: datetime | None = None, end: datetime | None = None, limit: int = 10):
    """Return recent reminders for a user matching optional keyword and date range."""
    pool = await get_pool()
    clauses = ["user_id = $1"]
    params: list = [user_id]
    idx = 2
    if keyword:
        clauses.append(f"reminder_text ILIKE ${idx}")
        params.append(f"%{keyword}%")
        idx += 1
    if start:
        clauses.append(f"created_at >= ${idx}")
        params.append(start)
        idx += 1
    if end:
        clauses.append(f"created_at <= ${idx}")
        params.append(end)
        idx += 1

    where_sql = " AND ".join(clauses)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT reminder_id, reminder_text, created_at, status
            FROM reminders
            WHERE {where_sql}
            ORDER BY created_at DESC
            LIMIT {limit}
            """,
            *params,
        )
    return [dict(r) for r in rows]

async def search_envelopes(user_id: str, keyword: str | None = None, start: datetime | None = None, end: datetime | None = None, limit: int = 10):
    """Return recent message envelopes for a user."""
    pool = await get_pool()
    clauses = ["user_id = $1"]
    params: list = [user_id]
    idx = 2
    if keyword:
        clauses.append(f"instruction ILIKE ${idx}")
        params.append(f"%{keyword}%")
        idx += 1
    if start:
        clauses.append(f"created_at >= ${idx}")
        params.append(start)
        idx += 1
    if end:
        clauses.append(f"created_at <= ${idx}")
        params.append(end)
        idx += 1

    where_sql = " AND ".join(clauses)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT envelope_id, instruction, created_at, payload
            FROM message_envelopes
            WHERE {where_sql}
            ORDER BY created_at DESC
            LIMIT {limit}
            """,
            *params,
        )
    return [dict(r) for r in rows]

# ──────────────────────────────────────────────
# New helpers
# ──────────────────────────────────────────────

async def update_envelope_status(envelope_id: str, status: str):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE message_envelopes
            SET status=$2, created_at=created_at -- touch nothing else
            WHERE envelope_id=$1
            """,
            envelope_id,
            status,
        )

async def claim_due_reminders(limit: int = 100):
    """Atomically set status='processing' and return claimed rows."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            WITH due AS (
                SELECT reminder_id FROM reminders
                WHERE status='pending' AND reminder_time <= NOW()
                ORDER BY reminder_time ASC
                LIMIT $1
            )
            UPDATE reminders r SET status='processing', updated_at=NOW()
            FROM due WHERE r.reminder_id = due.reminder_id
            RETURNING r.*
            """,
            limit,
        )
    return [dict(r) for r in rows]
