import pytest
from datetime import datetime, timezone, timedelta
from app.types.parser_contract import ReminderTask, TimeTrigger, ReminderReply


def test_time_trigger_validation():
    trigger = TimeTrigger(at=datetime.now(tz=timezone.utc) + timedelta(hours=1), timezone="UTC")
    task = ReminderTask(user_id="u1", reminder_text="hello", triggers=[trigger])
    reply = ReminderReply(need_clarification=False, reminder=task)
    assert not reply.need_clarification
    assert reply.reminder 