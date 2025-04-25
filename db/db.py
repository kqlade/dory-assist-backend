"""
Async DB helpers for the reminders MVP.
Uses SQLAlchemy 2.0 + asyncpg driver – no raw SQL strings in app code.
"""

from __future__ import annotations

import os
import json
import asyncio
from datetime import datetime, timezone
from typing import Sequence, TypedDict, Any
from uuid import uuid4

from sqlalchemy import (
    text, select, update, func
)
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, mapped_column
)
from sqlalchemy.ext.asyncio import (
    create_async_engine, async_sessionmaker, AsyncSession
)

from app.types.parser_contract import ReminderTask

# ──────────────────────────────────────────────────────────────────────
# 1. Engine & session factory
# ──────────────────────────────────────────────────────────────────────

DATABASE_URL = os.environ["DATABASE_PUBLIC_URL"].replace("postgres://", "postgresql+asyncpg://")

engine = create_async_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=5,
    pool_timeout=60,
)

Session: async_sessionmaker[AsyncSession] = async_sessionmaker(
    engine, expire_on_commit=False
)

# ──────────────────────────────────────────────────────────────────────
# 2. ORM models
# ──────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


class Reminder(Base):
    __tablename__ = "reminders"

    reminder_id: Mapped[str]   = mapped_column(primary_key=True)
    user_id:      Mapped[str]
    reminder_text:Mapped[str]
    channel:      Mapped[str]  = mapped_column(default="sms")
    status:       Mapped[str]  = mapped_column(default="pending")
    next_fire_at: Mapped[datetime | None]
    last_error:   Mapped[str | None]
    payload:      Mapped[dict[str, Any]]
    created_at:   Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at:   Mapped[datetime] = mapped_column(
        server_default=func.now(), onupdate=func.now()
    )


class MessageEnvelope(Base):
    __tablename__ = "message_envelopes"

    envelope_id: Mapped[str]   = mapped_column(primary_key=True)
    user_id:     Mapped[str | None]
    channel:     Mapped[str | None]
    instruction: Mapped[str]
    payload:     Mapped[dict[str, Any] | None]
    status:      Mapped[str]   = mapped_column(default="received")
    created_at:  Mapped[datetime] = mapped_column(server_default=func.now())


# ──────────────────────────────────────────────────────────────────────
# 3. DDL helper (run once at startup or from Alembic)
# ──────────────────────────────────────────────────────────────────────
async def create_all():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# ──────────────────────────────────────────────────────────────────────
# 4. CRUD helpers
# ──────────────────────────────────────────────────────────────────────

# 4.1 Insert envelope --------------------------------------------------
async def insert_envelope(envelope: dict):
    env = MessageEnvelope(
        envelope_id=envelope["envelope_id"],
        user_id=envelope.get("user_id"),
        channel=envelope.get("channel"),
        instruction=envelope["instruction"],
        payload=envelope.get("payload"),
        status=envelope.get("status", "received"),
        created_at=envelope.get("created_at", datetime.now(timezone.utc)),
    )
    async with Session() as s:
        s.add(env)
        await s.commit()


# 4.2 Insert reminder --------------------------------------------------
def _first_time_trigger(task: ReminderTask) -> datetime | None:
    for trg in task.triggers:
        if trg.type == "time":
            return trg.at
    return None


async def insert_reminder(task: ReminderTask) -> str:
    rid = str(uuid4())
    reminder = Reminder(
        reminder_id=rid,
        user_id=task.user_id,
        reminder_text=task.reminder_text,
        channel=task.channel,
        status="pending",
        next_fire_at=_first_time_trigger(task),
        payload=json.loads(task.model_dump_json()),
    )
    async with Session() as s:
        s.add(reminder)
        await s.commit()
    return rid


# 4.3 Claim due reminders ---------------------------------------------
async def claim_due_reminders(limit: int = 100) -> list[Reminder]:
    async with Session() as s:
        stmt = (
            update(Reminder)
            .where(
                Reminder.status == "pending",
                Reminder.next_fire_at <= func.now()
            )
            .values(status="processing", updated_at=func.now())
            .returning(Reminder)
            .order_by(Reminder.next_fire_at)
            .limit(limit)
        )
        res = await s.execute(stmt)
        await s.commit()
        return res.scalars().all()


# 4.4 mark_sent / mark_failed -----------------------------------------
async def mark_reminder_sent(rid: str):
    async with Session() as s:
        await s.execute(
            update(Reminder)
            .where(Reminder.reminder_id == rid)
            .values(status="sent", updated_at=func.now())
        )
        await s.commit()


async def mark_reminder_failed(rid: str, err: str):
    async with Session() as s:
        await s.execute(
            update(Reminder)
            .where(Reminder.reminder_id == rid)
            .values(status="failed", last_error=err, updated_at=func.now())
        )
        await s.commit()


# 4.5 Search helpers ---------------------------------------------------
async def search_reminders(
    user_id: str,
    keyword: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    limit: int = 10,
) -> list[dict]:
    async with Session() as s:
        stmt = select(Reminder).where(Reminder.user_id == user_id)

        if keyword:
            stmt = stmt.where(Reminder.reminder_text.ilike(f"%{keyword}%"))
        if start:
            stmt = stmt.where(Reminder.created_at >= start)
        if end:
            stmt = stmt.where(Reminder.created_at <= end)

        stmt = stmt.order_by(Reminder.created_at.desc()).limit(limit)

        res = await s.execute(stmt)
        return [r.__dict__ for r in res.scalars()]


async def search_envelopes(
    user_id: str,
    keyword: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    limit: int = 10,
) -> list[dict]:
    async with Session() as s:
        stmt = select(MessageEnvelope).where(MessageEnvelope.user_id == user_id)

        if keyword:
            stmt = stmt.where(MessageEnvelope.instruction.ilike(f"%{keyword}%"))
        if start:
            stmt = stmt.where(MessageEnvelope.created_at >= start)
        if end:
            stmt = stmt.where(MessageEnvelope.created_at <= end)

        stmt = stmt.order_by(MessageEnvelope.created_at.desc()).limit(limit)

        res = await s.execute(stmt)
        return [e.__dict__ for e in res.scalars()]


# 4.6 Envelope clarification helpers -----------------------------------
async def latest_awaiting_envelope(user_id: str) -> dict | None:
    async with Session() as s:
        stmt = (
            select(MessageEnvelope)
            .where(
                MessageEnvelope.user_id == user_id,
                MessageEnvelope.status == "awaiting_user",
            )
            .order_by(MessageEnvelope.created_at.desc())
            .limit(1)
        )
        res = await s.execute(stmt)
        env = res.scalar_one_or_none()
        return env.__dict__ if env else None


async def apply_clarification(envelope_id: str, new_instruction: str):
    async with Session() as s:
        await s.execute(
            update(MessageEnvelope)
            .where(MessageEnvelope.envelope_id == envelope_id)
            .values(instruction=new_instruction, status="received")
        )
        await s.commit()
