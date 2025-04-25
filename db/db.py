import os
import json
import asyncpg

from uuid import uuid4
from datetime import datetime, timezone

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
                created_at
            ) VALUES ($1, $2, $3, $4, $5::jsonb, $6)
            """,
            envelope["envelope_id"],
            envelope["user_id"],
            envelope["channel"],
            envelope["instruction"],
            json.dumps(envelope["payload"]),
            envelope["created_at"],
        )

# ──────────────────────────────────────────────
# Reminder helpers
# ──────────────────────────────────────────────

async def insert_reminder(task: ReminderTask):
    """Insert a pending reminder row and return its generated UUID."""

    reminder_id = str(uuid4())
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO reminders (
                reminder_id, user_id, reminder_text, reminder_time, timezone, channel,
                status, created_at, updated_at
            ) VALUES ($1, $2, $3, $4, $5, $6, 'pending', NOW(), NOW())
            """,
            reminder_id,
            task.user_id,
            task.reminder_text,
            task.reminder_time,
            task.timezone,
            task.channel,
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
                reminder_time TIMESTAMPTZ NOT NULL,
                timezone      TEXT NOT NULL,
                channel       TEXT NOT NULL DEFAULT 'sms',
                status        TEXT NOT NULL DEFAULT 'pending',
                last_error    TEXT,
                created_at    TIMESTAMPTZ DEFAULT NOW(),
                updated_at    TIMESTAMPTZ DEFAULT NOW()
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
                created_at   TIMESTAMPTZ DEFAULT now()
            );
            """
        )
