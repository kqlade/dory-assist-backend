import asyncio
import json

import pytest

from app.types.parser_contract import ReminderReply, ReminderTask, TimeTrigger
from app.services import parser_agent


def test_reminder_reply_round_trip():
    data = {
        "need_clarification": False,
        "reminder": {
            "user_id": "u1",
            "reminder_text": "Pay rent",
            "triggers": [
                {"type": "time", "at": "2030-01-01T09:00:00Z", "timezone": "UTC"}
            ],
        },
    }
    obj = ReminderReply.model_validate(data)
    cloned = ReminderReply.model_validate_json(obj.model_dump_json())
    assert cloned.reminder and cloned.reminder.reminder_text == "Pay rent"


@pytest.mark.asyncio
async def test_parser_agent_stub(monkeypatch):
    # Ensure OPENAI_API_KEY is empty so parser_agent returns stubbed reply
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    envelope = {
        "envelope_id": "e1",
        "user_id": "+1555",
        "channel": "sms",
        "instruction": "Save this",
        "payload": {"images": []},
        "created_at": "2025-01-01T00:00:00Z",
    }

    reply = await parser_agent.run(envelope, ocr_text=None)
    assert not reply.need_clarification
    assert isinstance(reply.reminder, ReminderTask)
