import os
import telnyx
import uuid
import datetime
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import PlainTextResponse
import db
from config import TELNYX_PUBLIC_KEY

# Configure telnyx public key
if TELNYX_PUBLIC_KEY:
    telnyx.public_key = TELNYX_PUBLIC_KEY

app = FastAPI()

# Create DB pool on startup and close on shutdown

@app.on_event("startup")
async def startup_event():
    await db.get_pool()

@app.on_event("shutdown")
async def shutdown_event():
    pool = await db.get_pool()
    await pool.close()

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
    except Exception:
        raise HTTPException(400, "Bad signature")

    # TelnyxObject -> dict if needed
    if hasattr(payload, "to_dict"):
        payload = payload.to_dict()

    from_num = payload.get("from", {}).get("phone_number")
    text     = payload.get("text", "")

    if not from_num:
        raise HTTPException(400, "Unexpected payload structure (missing from.phone_number)")

    # 2.1 Capture Telnyx media URLs directly (no re-hosting)
    images = [{"external_url": m["url"]} for m in payload.get("media", [])]

    # 3. Build and store your envelope row (replace the pseudo insert with your actual DB operation)
    envelope = {
        "envelope_id": str(uuid.uuid4()),
        "user_id": from_num,
        "channel": "sms",
        "instruction": text.strip(),
        "payload": {"images": images},
        "created_at": datetime.datetime.utcnow().isoformat()
    }
    # Store envelope in Postgres
    try:
        await db.insert_envelope(envelope)
    except Exception as e:
        # Log error but don't crash webhook
        print("DB insert failed:", e)
        print("Envelope:", envelope)
        raise HTTPException(status_code=500, detail="DB error")

    return "OK", status.HTTP_200_OK