import os
import telnyx
import uuid
import datetime
import asyncio
from fastapi import FastAPI, Request, HTTPException, status, BackgroundTasks
from fastapi.responses import PlainTextResponse
import db
from config import TELNYX_PUBLIC_KEY
from app.celery_app import celery_app
from app.services import parser_agent

# Background processing
from app.services import parser_agent

# Configure telnyx public key
if TELNYX_PUBLIC_KEY:
    telnyx.public_key = TELNYX_PUBLIC_KEY

app = FastAPI()

# Create DB pool on startup and close on shutdown

@app.on_event("startup")
async def startup_event():
    pass  # DB connections are now managed lazily
    # Tables now managed via Alembic migrations

@app.on_event("shutdown")
async def shutdown_event():
    await db.dispose_engine()

# --------------------------------------------
# Background task: parse envelope and enqueue
# --------------------------------------------

async def process_envelope_background(envelope: dict):
    """Parse envelope with LLM and push to appropriate queues."""
    try:
        reply = await parser_agent.run(envelope)
    except Exception as exc:
        # Don't bubble up; just log
        print("Parser agent failed:", exc)
        return

    if reply.need_clarification:
        # TODO: integrate SMS send & waiting state
        print("Need clarification:", reply.clarification_question)
        return

    # Reminder-only MVP: we expect a fully-populated reminder task
    if not reply.reminder:
        print("Parser returned no reminder and no clarification – skipping")
        return

    await db.insert_reminder(reply.reminder)

    # Log helpful debug of first time trigger if available
    first_time = None
    for trig in reply.reminder.triggers:
        if getattr(trig, "type", None) == "time":
            first_time = getattr(trig, "at", None)
            break
    print("Reminder stored", first_time or "(non-time trigger)")

# --------------------------------------------
# Endpoint
# --------------------------------------------
@app.post("/v1/sms/telnyx", response_class=PlainTextResponse)
async def telnyx_webhook(request: Request, background: BackgroundTasks):
    raw_body  = await request.body()
    sig = request.headers.get("telnyx-signature-ed25519")
    ts  = request.headers.get("telnyx-timestamp")

    print("[Webhook] Raw incoming payload:", raw_body)
    try:
        if TELNYX_PUBLIC_KEY:
            event = telnyx.Webhook.construct_event(
                raw_body.decode(), sig, ts
            )
            payload = event.data["payload"]
        else:  # dev mode: skip signature verification
            payload = (await request.json())["data"]["payload"]
    except Exception as e:
        raise HTTPException(400, "Bad signature")

    if payload.get("type") == "ping":
        return PlainTextResponse("PONG")

    # TelnyxObject -> dict if needed
    if hasattr(payload, "to_dict"):
        payload = payload.to_dict()

    sender = payload.get("from") or payload.get("from_", {})
    if hasattr(sender, "to_dict"):
        sender = sender.to_dict()
    from_num = sender.get("phone_number")
    text     = payload.get("text", "")

    if not from_num:
        return PlainTextResponse("IGNORED", status_code=status.HTTP_200_OK)

    images = [{"external_url": m["url"]} for m in payload.get("media", [])]

    awaiting = await db.latest_awaiting_envelope(from_num)
    if awaiting:
        merged_instruction = f"{awaiting['instruction']} {text.strip()}".strip()
        updated = await db.apply_clarification(awaiting["envelope_id"], merged_instruction)
        background.add_task(parser_agent.run_and_store, updated)
        return PlainTextResponse("CLARIFICATION_RECEIVED", status_code=200)

    UTC = datetime.timezone.utc
    # Derive logical timezone string; fallback to 'UTC'.
    created_at = datetime.datetime.now(tz=UTC)
    tz_str = created_at.tzname() or "UTC"
    envelope = {
        "envelope_id": str(uuid.uuid4()),
        "user_id": from_num,
        "channel": "sms",
        "instruction": text.strip(),
        "payload": {"images": images},
        "created_at": created_at,
        "timezone": tz_str,
    }
    print("[Webhook] Constructed envelope:", envelope)
    try:
        print("[Webhook] Inserting envelope into DB...")
        await db.insert_envelope(envelope)
        print("[Webhook] Added background task for parser_agent.run_and_store")
        background.add_task(parser_agent.run_and_store, envelope)
    except Exception as e:
        print("DB insert failed:", e)
        raise HTTPException(500, "DB error")
    return PlainTextResponse("OK")