from __future__ import annotations

"""Periodic scanner to send due reminders.
Run via Railway schedule every minute:
    python -m app.scripts.scan_due_reminders
"""

import asyncio
import os

import telnyx

from app.utils.sms import send_sms
import db

TELNYX_PUBLIC_KEY = os.getenv("TELNYX_PUBLIC_KEY")
if TELNYX_PUBLIC_KEY:
    telnyx.api_key = TELNYX_PUBLIC_KEY


async def main() -> None:
    due = await db.fetch_due_reminders()
    if not due:
        return

    for row in due:
        try:
            await _process_row(row)
        except Exception as e:  # noqa: BLE001
            await db.mark_reminder_failed(row["reminder_id"], str(e))
            print("Failed to send reminder", row["reminder_id"], e)


async def _process_row(row: dict) -> None:
    # Send SMS (future: other channels)
    send_sms(row["user_id"], row["reminder_text"])
    await db.mark_reminder_sent(row["reminder_id"])
    print("Reminder sent", row["reminder_id"])


if __name__ == "__main__":  # pragma: no cover
    print("[CRON] scan_due_reminders: job started")
    try:
        asyncio.run(main())
        print("[CRON] scan_due_reminders: job completed successfully")
    except Exception as e:
        print(f"[CRON] scan_due_reminders: job failed: {e}") 