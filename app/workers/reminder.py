"""Placeholder reminder task."""

from __future__ import annotations

from app.celery_app import celery_app
from app.utils.sms import send_sms
import asyncio
import db


# ---------------------------------------------------------------------------
# Celery Tasks
# ---------------------------------------------------------------------------

@celery_app.task(name="app.workers.reminder.handle", bind=True, max_retries=3)
def handle(self, reminder_id: str, user_id: str, reminder_text: str):  # noqa: D401
    """Send a single reminder SMS and mark DB status accordingly."""
    try:
        send_sms(user_id, reminder_text)
        # Mark sent in DB (async helper wrapped via asyncio)
        asyncio.run(db.mark_reminder_sent(reminder_id))
    except Exception as exc:  # noqa: BLE001
        # Mark failure + retry with backoff
        asyncio.run(db.mark_reminder_failed(reminder_id, str(exc)))
        raise self.retry(exc=exc)


@celery_app.task(name="app.workers.reminder.dispatch_due", bind=True)
def dispatch_due(self):  # noqa: D401
    """Fetch due reminders and enqueue handle tasks for each."""
    try:
        due: list[dict] = asyncio.run(db.fetch_due_reminders(limit=100))
    except Exception as exc:  # noqa: BLE001
        self.retry(exc=exc, countdown=30)
        return

    for row in due:
        celery_app.send_task(
            "app.workers.reminder.handle",
            args=[row["reminder_id"], row["user_id"], row["reminder_text"]],
            queue="reminder",
        )
