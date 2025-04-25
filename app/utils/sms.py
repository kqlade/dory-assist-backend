import os
import telnyx

FROM_NUM = os.getenv("TELNYX_FROM")
TELNYX_API_KEY = os.getenv("TELNYX_API_KEY")
if TELNYX_API_KEY:
    telnyx.api_key = TELNYX_API_KEY

def send_sms(to: str, body: str) -> None:
    if not TELNYX_API_KEY or not FROM_NUM:
        print("[SMS] DEV mode: would send to", to, ":", body)
        return
    telnyx.Message.create(from_=FROM_NUM, to=to, text=body) 