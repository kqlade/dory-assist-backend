import pytest, asyncio
from app.workers import reminder as reminder_worker
from app.utils import sms as sms_util
import db
from datetime import datetime, timezone, timedelta
from app.types.parser_contract import ReminderTask, TimeTrigger
import os


@pytest.mark.asyncio
async def test_dispatch_and_handle(monkeypatch):
    sent_messages = []

    def fake_send_sms(to, body):
        sent_messages.append((to, body))

    monkeypatch.setattr(sms_util, "send_sms", fake_send_sms)

    # Ensure DB env var set for tests (in-memory SQLite not available, so skip if no DATABASE_URL)
    if not (db_url := os.getenv("DATABASE_URL")):
        pytest.skip("DATABASE_URL not set for reminder worker test")

    # Insert a reminder due now
    trigger = TimeTrigger(at=datetime.now(tz=timezone.utc) - timedelta(minutes=1), timezone="UTC")
    task = ReminderTask(user_id="+10000000000", reminder_text="test msg", triggers=[trigger])
    await db.insert_reminder(task)

    # Run dispatch synchronously
    reminder_worker.dispatch_due.apply(args=())

    assert sent_messages, "SMS should have been sent" 