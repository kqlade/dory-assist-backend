import os
import json
import asyncpg

_pool = None

async def get_pool():
    """Get (or create) a global asyncpg pool."""
    global _pool
    if _pool is None:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise RuntimeError("DATABASE_URL environment variable is not set")
        _pool = await asyncpg.create_pool(database_url, min_size=1, max_size=5)
    return _pool

async def insert_envelope(envelope: dict):
    """Insert an envelope row into message_envelopes table."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO message_envelopes (
                envelope_id,
                user_id,
                channel,
                instruction,
                payload,
                created_at
            ) VALUES ($1, $2, $3, $4, $5::jsonb, $6)
            """,
            envelope["envelope_id"],
            envelope["user_id"],
            envelope["channel"],
            envelope["instruction"],
            json.dumps(envelope["payload"]),
            envelope["created_at"],
        )
