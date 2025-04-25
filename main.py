import os
import telnyx
import uuid
import datetime
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import PlainTextResponse

from config import TELNYX_PUBLIC_KEY

# Configure telnyx public key
if TELNYX_PUBLIC_KEY:
    telnyx.public_key = TELNYX_PUBLIC_KEY

app = FastAPI()

@app.post("/v1/sms/telnyx", response_class=PlainTextResponse)
async def telnyx_webhook(request: Request):
    raw_body  = await request.body()
    signature = request.headers.get("telnyx-signature-ed25519")
    ts        = request.headers.get("telnyx-timestamp")

    # 1. Verify webhook authenticity (raises if bad)
    try:
        telnyx.verify_webhook_signature(raw_body, signature, ts)
    except telnyx.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Bad signature")

    # 2. Extract sender & text (Telnyx v2 JSON payload structure)
    payload = (await request.json())["data"]["payload"]
    from_num = payload["from"]["phone_number"]
    text     = payload["text"]

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
    # Replace this with your actual DB insert call
    # db.insert_envelope(envelope)
    print("Storing envelope:", envelope)

    return "OK", status.HTTP_200_OK