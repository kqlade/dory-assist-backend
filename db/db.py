"""
Async DB helpers for the reminders MVP.
Uses SQLAlchemy 2.0 + asyncpg driver – no raw SQL strings in app code.
"""

from __future__ import annotations

import os
import json
import asyncio
from datetime import datetime, timezone
from typing import Sequence, TypedDict, Any, AsyncGenerator
from uuid import uuid4

from sqlalchemy import (
    text, select, update, func, JSON, DateTime
)
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, mapped_column
)
from sqlalchemy.ext.asyncio import (
    create_async_engine, async_sessionmaker, AsyncSession
)

from app.types.parser_contract import ReminderTask

# ──────────────────────────────────────────────────────────────────────
# 1. Declarative metadata
# ──────────────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass

# ──────────────────────────────────────────────────────────────────────
# 2. Lazy engine / session factory
# ──────────────────────────────────────────────────────────────────────
_engine = None
_session_maker: async_sessionmaker[AsyncSession] | None = None

def _build_url() -> str:
    url = os.getenv("DATABASE_URL") or os.getenv("DATABASE_PUBLIC_URL")
    if not url:
        raise RuntimeError("DATABASE_URL not set")
    if "+asyncpg" not in url:
        url = url.replace("postgres://", "postgresql+asyncpg://", 1).replace(
            "postgresql://", "postgresql+asyncpg://", 1
        )
    return url

def get_engine():
    global _engine
    if _engine is None:
        _engine = create_async_engine(_build_url(), pool_size=5, max_overflow=5)
    return _engine

def get_session() -> AsyncGenerator[AsyncSession, None]:
    global _session_maker
    if _session_maker is None:
        _session_maker = async_sessionmaker(get_engine(), expire_on_commit=False)
    async def _session_scope():
        async with _session_maker() as session:
            yield session
    return _session_scope()

# ──────────────────────────────────────────────────────────────────────
# 3. ORM models
# ──────────────────────────────────────────────────────────────────────

class Reminder(Base):
    __tablename__ = "reminders"

    reminder_id: Mapped[str]   = mapped_column(primary_key=True)
    user_id:      Mapped[str]
    reminder_text:Mapped[str]
    channel:      Mapped[str]  = mapped_column(default="sms")
    status:       Mapped[str]  = mapped_column(default="pending")
    next_fire_at: Mapped[datetime | None]
    last_error:   Mapped[str | None]
    payload:      Mapped[dict[str, Any]] = mapped_column(JSON)
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
    payload:     Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    status:      Mapped[str]   = mapped_column(default="received")
    created_at:  Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# ──────────────────────────────────────────────────────────────────────
# 4. DDL helper (run once at startup or from Alembic)
# ──────────────────────────────────────────────────────────────────────
async def create_all():
    async with get_engine().begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# ──────────────────────────────────────────────────────────────────────
# 5. CRUD helpers
# ──────────────────────────────────────────────────────────────────────

# 5.1 Insert envelope --------------------------------------------------
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
    async for s in get_session():
        s.add(env)
        await s.commit()


# 5.2 Insert reminder --------------------------------------------------
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
    async for s in get_session():
        s.add(reminder)
        await s.commit()
    return rid


# 5.3 Claim due reminders ---------------------------------------------
async def claim_due_reminders(limit: int = 100) -> list[Reminder]:
    async for s in get_session():
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


# 5.4 mark_sent / mark_failed -----------------------------------------
async def mark_reminder_sent(rid: str):
    async for s in get_session():
        await s.execute(
            update(Reminder)
            .where(Reminder.reminder_id == rid)
            .values(status="sent", updated_at=func.now())
        )
        await s.commit()


async def mark_reminder_failed(rid: str, err: str):
    async for s in get_session():
        await s.execute(
            update(Reminder)
            .where(Reminder.reminder_id == rid)
            .values(status="failed", last_error=err, updated_at=func.now())
        )
        await s.commit()


# 5.5 Search helpers ---------------------------------------------------
async def search_reminders(
    user_id: str,
    keyword: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    limit: int = 10,
) -> list[dict]:
    async for s in get_session():
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
    async for s in get_session():
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


# 5.6 Envelope clarification helpers -----------------------------------
async def latest_awaiting_envelope(user_id: str) -> dict | None:
    async for s in get_session():
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
    async for s in get_session():
        await s.execute(
            update(MessageEnvelope)
            .where(MessageEnvelope.envelope_id == envelope_id)
            .values(instruction=new_instruction, status="received")
        )
        await s.commit()


async def dispose_engine():
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None
