from datetime import datetime
import pytest
import asyncio
from db.db import insert_envelope

@pytest.mark.asyncio
async def test_insert_naive_datetime_raises():
    envelope = {
        "envelope_id": "test-uuid",
        "user_id": "+1234567890",
        "channel": "sms",
        "instruction": "test",
        "payload": {},
        "created_at": datetime(2025, 4, 25, 15, 0, 0),  # Naive!
    }
    with pytest.raises(ValueError, match="timezone-aware"):
        await insert_envelope(envelope)

@pytest.mark.asyncio
async def test_insert_aware_datetime_succeeds():
    envelope = {
        "envelope_id": "test-uuid-2",
        "user_id": "+1234567890",
        "channel": "sms",
        "instruction": "test",
        "payload": {},
        "created_at": datetime(2025, 4, 25, 15, 0, 0, tzinfo=datetime.timezone.utc),
    }
    # Should not raise
    await insert_envelope(envelope)
