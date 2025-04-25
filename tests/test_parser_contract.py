import asyncio
import json

import pytest

from app.types.parser_contract import ParserReply, EntityDraft
from app.services import parser_agent


def test_parser_reply_round_trip():
    data = {
        "intent": "save",
        "confidence": 0.85,
        "need_clarification": False,
        "entities": [
            {
                "type": "place",
                "name": "Chez Janou",
                "city": "Paris",
                "tags": ["Restaurant", "France"],
                "needs_resolution": True,
            }
        ],
    }
    obj = ParserReply.parse_obj(data)
    cloned = ParserReply.parse_raw(obj.json())
    assert cloned.intent == "save"
    assert cloned.entities[0].tags == ["restaurant", "france"]


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
    assert reply.intent == "save"
    assert reply.confidence > 0.0
