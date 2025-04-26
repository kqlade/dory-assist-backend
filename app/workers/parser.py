"""Celery worker that parses an envelope using the LLM agent."""

from __future__ import annotations

import asyncio
from typing import Dict

from app.celery_app import celery_app
from app.services import parser_agent
from app.utils.sms import send_sms
import db


@celery_app.task(name="app.workers.parser.handle_envelope", bind=True, max_retries=3)
def handle_envelope(self, envelope: Dict):
    """Parse envelope and fan-out to downstream queues/storage."""

    try:
        print("[Worker] Calling parser_agent.run with envelope:", envelope)
        reply = asyncio.run(parser_agent.run(envelope))
        print("[Worker] parser_agent.run returned:", reply)
    except Exception as exc:
        print("[Worker] Exception in parser_agent.run:", exc)
        raise self.retry(exc=exc, countdown=30)

    user_id = envelope.get("user_id", "")

    if reply.need_clarification:
        question = reply.clarification_question or "Could you clarify your request?"
        print(f"[Worker] Sending clarification SMS to {user_id}: {question}")
        send_sms(user_id, question)
        print(f"[Worker] Updating envelope status to 'awaiting_user' for {envelope['envelope_id']}")
        asyncio.run(db.update_envelope_status(envelope["envelope_id"], "awaiting_user"))
        return

    if not reply.reminder:
        print("[Worker] No reminder produced for envelope", envelope.get("envelope_id"))
        print(f"[Worker] Updating envelope status to 'no_reminder' for {envelope['envelope_id']}")
        asyncio.run(db.update_envelope_status(envelope["envelope_id"], "no_reminder"))
        return

    # Store reminder and send confirmation
    try:
        print(f"[Worker] Inserting reminder into DB for user {user_id}: {reply.reminder}")
        asyncio.run(db.insert_reminder(reply.reminder))
    except Exception as exc:
        print("[Worker] Exception inserting reminder:", exc)
        raise self.retry(exc=exc, countdown=30)

    print(f"[Worker] Sending confirmation SMS to {user_id}: Got it! I'll remind you: {reply.reminder.reminder_text}")
    send_sms(user_id, f"Got it! I'll remind you: {reply.reminder.reminder_text}")
    print(f"[Worker] Updating envelope status to 'parsed' for {envelope['envelope_id']}")
    asyncio.run(db.update_envelope_status(envelope["envelope_id"], "parsed"))