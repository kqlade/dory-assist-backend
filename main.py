import os
import telnyx
import uuid
import datetime
import asyncio
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import PlainTextResponse
import db
from config import TELNYX_PUBLIC_KEY
from app.celery_app import celery_app

# Background processing
from app.services import parser_agent

# Configure telnyx public key
if TELNYX_PUBLIC_KEY:
    telnyx.public_key = TELNYX_PUBLIC_KEY

app = FastAPI()

# Create DB pool on startup and close on shutdown

@app.on_event("startup")
async def startup_event():
    await db.get_pool()
    await db.create_message_envelopes_table()
    await db.create_reminders_table()

@app.on_event("shutdown")
async def shutdown_event():
    pool = await db.get_pool()
    await pool.close()

# --------------------------------------------
# Background task: parse envelope and enqueue
# --------------------------------------------

async def process_envelope_background(envelope: dict):
    """Parse envelope with LLM and push to appropriate queues."""
    try:
        reply = await parser_agent.run(envelope, ocr_text=None)
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
    print("Reminder stored", reply.reminder.reminder_time)

@app.post("/v1/sms/telnyx", response_class=PlainTextResponse)
async def telnyx_webhook(request: Request):
    raw_body  = await request.body()
    sig = request.headers.get("telnyx-signature-ed25519")
    ts  = request.headers.get("telnyx-timestamp")

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

    # TelnyxObject -> dict if needed
    if hasattr(payload, "to_dict"):
        payload = payload.to_dict()

    sender = payload.get("from") or payload.get("from_", {})
    if hasattr(sender, "to_dict"):
        sender = sender.to_dict()
    from_num = sender.get("phone_number")
    text     = payload.get("text", "")

    if not from_num:
        # Not an inbound message we care about (e.g., DLR). Acknowledge and exit.
        return PlainTextResponse("IGNORED", status_code=status.HTTP_200_OK)

    # 2.1 Capture Telnyx media URLs directly (no re-hosting)
    images = [{"external_url": m["url"]} for m in payload.get("media", [])]

    # 3. Build and store your envelope row (replace the pseudo insert with your actual DB operation)
    UTC = datetime.timezone.utc
    envelope = {
        "envelope_id": str(uuid.uuid4()),
        "user_id": from_num,
        "channel": "sms",
        "instruction": text.strip(),
        "payload": {"images": images},
        "created_at": datetime.datetime.now(tz=UTC)
    }
    # Store envelope in Postgres
    try:
        await db.insert_envelope(envelope)
        # Fire-and-forget background processing
        asyncio.create_task(process_envelope_background(envelope))
    except Exception as e:
        # Log error but don't crash webhook
        print("DB insert failed:", e)
        print("Envelope:", envelope)
        raise HTTPException(status_code=500, detail="DB error")

    return PlainTextResponse("OK", status_code=status.HTTP_200_OK)