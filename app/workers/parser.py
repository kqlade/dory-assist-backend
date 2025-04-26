"""Celery worker that parses an envelope using the LLM agent.

Flow:
1. Receive raw `envelope` dict saved by the webhook.
2. Call `parser_agent.run` (async) to extract a `ReminderReply`.
3. • If clarification is required → send an SMS question to the user.
   • Else → store the `ReminderTask` in the DB and (optionally) confirm to user.

Retries: Any exception bubbles up to `self.retry` so failures are retried with
exponential back-off by Celery.
"""
from __future__ import annotations

import asyncio
from typing import Dict

from app.celery_app import celery_app
from app.services import parser_agent
from app.utils.sms import send_sms
import db


@celery_app.task(name="app.workers.parser.handle_envelope", bind=True, max_retries=3)
def handle_envelope(self, envelope: Dict):  # noqa: D401, ANN401
    """Parse envelope and fan-out to downstream queues/storage.

    This task is defined as a *synchronous* function so that it runs correctly
    with Celery's default prefork pool.  We run any async code using
    ``asyncio.run`` to avoid returning coroutine objects (which cannot be
    JSON-serialised by Kombu).
    """

    try:
        # parser_agent.run is async – execute synchronously inside the worker
        reply = asyncio.run(parser_agent.run(envelope))
    except Exception as exc:  # noqa: BLE001
        # Retry with back-off so transient LLM errors don't lose the job
        raise self.retry(exc=exc, countdown=30)

    user_id = envelope.get("user_id", "")

    if reply.need_clarification:
        question = reply.clarification_question or "Could you clarify your request?"
        send_sms(user_id, question)
        # Mark status so webhook can re-process reply later
        asyncio.run(db.update_envelope_status(envelope["envelope_id"], "awaiting_user"))
        return

    if not reply.reminder:
        # No useful output
        print("[ParserWorker] No reminder produced for envelope", envelope.get("envelope_id"))
        asyncio.run(db.update_envelope_status(envelope["envelope_id"], "no_reminder"))
        return

    # Store reminder and send confirmation
    try:
        asyncio.run(db.insert_reminder(reply.reminder))
    except Exception as exc:  # noqa: BLE001
        raise self.retry(exc=exc, countdown=30)

    # Confirmation to user (can be toggled via env var in future)
    send_sms(user_id, f"Got it! I'll remind you: {reply.reminder.reminder_text}")
    # Update envelope status to parsed
    asyncio.run(db.update_envelope_status(envelope["envelope_id"], "parsed"))
